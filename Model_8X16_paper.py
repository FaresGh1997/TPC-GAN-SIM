import tensorflow as tf 
import numpy as np
import pandas as pd
import math
import re
from tqdm import trange
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
from metrics import make_images_for_model,evaluate_model
import h5py
from tensorflow.python.keras.saving import hdf5_format


_THIS_PATH = Path(os.path.realpath(__file__)).parent




def setup_gpu():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    assert len(logical_devices) > 0, "Not enough GPU hardware devices available"


# function to convert raw csv data into data and corresponding features for the model
def read_csv_2d(filename=None, pad_range=(40, 50), time_range=(265, 280), strict=True):
    if filename is None:
        filename = str(_THIS_PATH.joinpath('Data', 'digits.csv'))

    df = pd.read_csv(filename)

    sel = lambda df, col, limits: (df[col] >= limits[0]) & (df[col] < limits[1])


    if 'drift_length' in df.columns:
        df['itime'] -= df['drift_length'].astype(int)

    if 'pad_coordinate' in df.columns:
        df['ipad'] -= df['pad_coordinate'].astype(int)

    selection = sel(df, 'itime', time_range) & sel(df, 'ipad', pad_range)
    
    if not selection.all():
        msg = "WARNING: current selection ignores {value} of the data!".format(value = (~selection).sum() / len(selection) * 100)
        assert not strict, msg
        print(msg)

    g = df[selection].groupby('evtId')
    
    def convert_event(event):
        result = np.zeros(dtype=float, shape=(pad_range[1] - pad_range[0], time_range[1] - time_range[0]))

        indices = tuple(event[['ipad', 'itime']].values.T - np.array([[pad_range[0]], [time_range[0]]]))
        result[indices] = event.amp.values

        return result
    
    data = np.stack(g.apply(convert_event).values)
    
    if 'crossing_angle' in df.columns:
        features = ['crossing_angle', 'dip_angle']
        if 'drift_length' in df.columns:
            features += ['drift_length']
        if 'pad_coordinate' in df.columns:
            features += ['pad_coordinate']
        assert (g[features].std() == 0).all().all(), 'Varying features within same events...'
        return data, g[features].mean().values
    
    return data

    

#
@tf.function
def preprocessing_func_as_tensor(input_tensors):
    bin_fractions = input_tensors[:, 2:4] % 1
    input_tensors = (input_tensors[:, :3] - tf.constant([[0.0, 0.0, 162.5]])) / tf.constant([[20.0, 60.0, 127.5]])
    return tf.concat([input_tensors, bin_fractions], axis=-1)


_f = preprocessing_func_as_tensor
def custom_activation_V2 (x, shift = 0.01,val = np.log10(2),val0 = np.log10(2) / 10):
    return tf.where(x> shift,
                    val +x -shift,
                    val0 + tf.keras.activations.elu(
                        x,alpha = (val0 *shift/ (val -val0))
                                                    ) * (val -val0)/shift)
    
def create_generator_structure():
    
    generator = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32,input_shape = (37,) , activation='elu', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dense(64,activation='elu', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dense(64,activation='elu', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dense(64,activation='elu', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dense(8*16,activation=custom_activation_V2, kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Reshape(target_shape= (8,16))
        ],
        name= 'generator'
    )
    return generator

#Appendix A: discriminator

def create_discriminator_structure():
    
    discriminator_main = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(activation='elu', kernel_size=(3,3),filters= 16 , padding='same', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dropout(0.02),
            
            tf.keras.layers.Conv2D(activation='elu',kernel_size= (3,3),filters=16, padding='same', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dropout(0.02),
            
            tf.keras.layers.MaxPool2D(pool_size=(1,2)),
            
            tf.keras.layers.Conv2D(activation='elu', kernel_size=(3,3),filters= 32 , padding='same', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dropout(0.02),
            
            tf.keras.layers.Conv2D(activation='elu',kernel_size= (3,3),filters=32, padding='same', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dropout(0.02),
            
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            
            tf.keras.layers.Conv2D(activation='elu', kernel_size=(3,3),filters= 64 , padding='valid', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dropout(0.02),
            
            tf.keras.layers.Conv2D(activation='elu',kernel_size= (2,2),filters=64, padding='valid', kernel_initializer= 'glorot_uniform'),
            tf.keras.layers.Dropout(0.02),
            
            tf.keras.layers.Reshape(target_shape=(64,))
        ],
        name= 'discriminator_main'
    )
    
    vector_shape = tuple([5,])
    img_shape = tuple([8, 16])
    
    input_vec = tf.keras.Input(shape=vector_shape)
    input_img = tf.keras.Input(shape=img_shape)
    
    block_input = input_img
    
    if len(img_shape) == 2:
        block_input = tf.keras.layers.Reshape(img_shape + (1,))(block_input)
       
    reshaped_vec = tf.tile(
    tf.keras.layers.Reshape((1, 1) + vector_shape)(input_vec),
        (1, img_shape[0],img_shape[1], 1)
    )
    block_input = tf.keras.layers.Concatenate(axis=-1)([block_input, reshaped_vec])

    block_output = discriminator_main(block_input)
    outputs = [input_vec, block_output]
    
    outputs = tf.keras.layers.Concatenate(axis=-1)(outputs)
    
    discriminator_tail = tf.keras.Model(inputs = [input_vec,input_img], outputs = outputs , name = "discriminator_tail")
    
    
    
    
    discriminator_head = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape = (69,),activation='elu', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(1,activation=None)
        ], name= 'discriminator_head'
        
    )
    
    blocks = [discriminator_tail ,discriminator_head]
    
    inputs = [
        tf.keras.Input(shape = i.shape[1:]) for i in blocks[0].inputs
    ]
    outputs = inputs
    
    for block in blocks:
        outputs = block(outputs)
        
        
    args = dict(
        inputs=inputs,
        outputs=outputs,
        name = 'discriminator'
    )
    
    return tf.keras.Model(**args)
    
def gen_loss (fake,real):
    #return tf.reduce_mean (real) - tf.reduce_mean(fake)
    return tf.reduce_mean (real - fake)

def disc_loss (fake, real):
    #return tf.reduce_mean (real) - tf.reduce_mean(fake)
    return tf.reduce_mean (fake - real)
   

class GAN_Model:
    def __init__(self, latent_dim = 32 , batch_size  = 32, learning_rate = 1.e-4,
                 learning_schedule =0.999,lambda_value = 10 , num_update_step = 8,
                 num_epochs = 10000):
        self.preprocess_function = preprocessing_func_as_tensor
        
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # We train both generator and discriminator using RMSprop optimizer with learning rates starting at 0.0001
        
        self.lr = learning_rate
        self.lr_schedule = learning_schedule
        self.disc_opt = tf.keras.optimizers.RMSprop(learning_rate= self.lr)
        self.gen_opt = tf.keras.optimizers.RMSprop (learning_rate= self.lr)
        
        # gradient penalty value
        self.lambda_value = lambda_value
        
        #We make 8 discriminator update steps per single generator step.
        self.num_update_step =  num_update_step
        
        self.num_epochs = num_epochs
        
        self.generator = create_generator_structure()
        self.discriminator = create_discriminator_structure()
        
        
        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)
        self.scaler = Logarithmic()
        
        self.generator.compile(optimizer=self.gen_opt, loss='mean_squared_error')
        self.discriminator.compile(optimizer=self.disc_opt, loss='mean_squared_error')
    
    def load_generator(self, checkpoint):
        self._load_weights(checkpoint, 'gen')

    def load_discriminator(self, checkpoint):
        self._load_weights(checkpoint, 'disc')

    def _load_weights(self, checkpoint, gen_or_disc):
        if gen_or_disc == 'gen':
            network = self.generator
            step_fn = self.gen_step
        elif gen_or_disc == 'disc':
            network = self.discriminator
            step_fn = self.disc_step
        else:
            raise ValueError(gen_or_disc)

        model_file = h5py.File(checkpoint, 'r')
        if len(network.optimizer.weights) == 0 and 'optimizer_weights' in model_file:
            # perform single optimization step to init optimizer weights
            features_shape = self.discriminator.inputs[0].shape.as_list()
            targets_shape = self.discriminator.inputs[1].shape.as_list()
            features_shape[0], targets_shape[0] = 1, 1
            step_fn(tf.zeros(features_shape), tf.zeros(targets_shape))

        print(f'Loading {gen_or_disc} weights from {str(checkpoint)}')
        network.load_weights(str(checkpoint))
        
        if 'optimizer_weights' in model_file:
            print('Also recovering the optimizer state')
            opt_weight_values = hdf5_format.load_optimizer_weights_from_hdf5_group(model_file)
            network.optimizer.set_weights(opt_weight_values)    
        
    @tf.function
    def make_fake (self,features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size,self.latent_dim),dtype='float32')
        return self.generator(
            tf.concat([_f(features), latent_input],axis= -1)
            )
    
    def gradient_penalty(self, features, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1])
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator([_f(features), interpolates])
            #print ("we are pass this ")
        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0) ** 2)
    
    @tf.function
    def calculate_losses(self, feature_batch, target_batch):
        fake = self.make_fake(feature_batch)
        
        #print ("batch data shape  inside calculating loss= ", target_batch.shape)
        #print ("batch features shape inside calculating loss = ", feature_batch.shape)
        #print ("batch features shape after preprocessing = ", _f(feature_batch).shape)
        #print ("fake shape = ",fake.shape)
        d_real = self.discriminator([_f(feature_batch), target_batch])
        
        d_fake = self.discriminator([_f(feature_batch), fake])
        
        # calculating loss
        d_loss = disc_loss(d_real, d_fake)
        # adding Gradient Penalty 
        d_loss = d_loss + self.gradient_penalty(feature_batch, target_batch, fake) * self.lambda_value
        # calculating generator loss
        g_loss = gen_loss(d_real, d_fake)
    
        return {'disc_loss': d_loss, 'gen_loss': g_loss}
        
    def disc_step(self, feature_batch, target_batch):
        feature_batch = tf.convert_to_tensor(feature_batch)
        target_batch = tf.convert_to_tensor(target_batch)
        
        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        grads = t.gradient(losses['disc_loss'], self.discriminator.trainable_variables)
        self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return losses
    
    def gen_step(self, feature_batch, target_batch):
        feature_batch = tf.convert_to_tensor(feature_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        grads = t.gradient(losses['gen_loss'], self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        return losses
    
    
    @tf.function
    def training_step(self, feature_batch, target_batch):
        if self.step_counter == self.num_update_step:
            result = self.gen_step(feature_batch, target_batch)
            self.step_counter.assign(0)
        else:
            result = self.disc_step(feature_batch, target_batch)
            self.step_counter.assign_add(1)
        return result


def epoch_from_name(name):
    epoch, = re.findall('\d+', name)
    return int(epoch)


def latest_epoch(model_path):
    gen_checkpoints = model_path.glob("generator_*.h5")
    disc_checkpoints = model_path.glob("discriminator_*.h5")

    gen_epochs = [epoch_from_name(path.stem) for path in gen_checkpoints]
    disc_epochs = [epoch_from_name(path.stem) for path in disc_checkpoints]

    latest_gen_epoch = max(gen_epochs)
    latest_disc_epoch = max(disc_epochs)

    assert latest_gen_epoch == latest_disc_epoch, "Latest disc and gen epochs differ"

    return latest_gen_epoch

def load_weights(model, model_path, epoch=None):
    if epoch is None:
        epoch = latest_epoch(model_path)

    gen_checkpoint = model_path / f"generator_{epoch:05d}.h5"
    disc_checkpoint = model_path / f"discriminator_{epoch:05d}.h5"

    model.load_generator(gen_checkpoint)
    model.load_discriminator(disc_checkpoint)

    return epoch

class SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.path = path
        self.save_period = save_period

    def __call__(self, step):
        if step % self.save_period == 0:
            print('Saving model on step {s} to {p}'.format(s = step, p = self.path))
            self.model.generator.save(str(self.path.joinpath("generator_{:05d}.h5".format(step))))
            self.model.discriminator.save(str(self.path.joinpath("discriminator_{:05d}.h5".format(step))))


class WriteHistSummaryCallback:
    def __init__(self, model, sample, save_period, writer):
        self.model = model
        self.sample = sample
        self.save_period = save_period
        self.writer = writer

    def __call__(self, step):
        if step % self.save_period == 0:
            images, images1, img_amplitude, chi2 = make_images_for_model(self.model,
                                                              sample=self.sample,
                                                              calc_chi2=True)
            with self.writer.as_default():
                tf.summary.scalar("chi2", chi2, step)

                for k, img in images.items():
                    tf.summary.image(k, img, step)
                for k, img in images1.items():
                    tf.summary.image("{} (amp > 1)".format(k), img, step)
                tf.summary.image("log10(amplitude + 1)", img_amplitude, step)


class ScheduleLRCallback:
    def __init__(self, model, func_gen, func_disc, writer):
        self.model = model
        self.func_gen = func_gen
        self.func_disc = func_disc
        self.writer = writer

    def __call__(self, step):
        self.model.disc_opt.lr.assign(self.func_disc(step))
        self.model.gen_opt.lr.assign(self.func_gen(step))
        with self.writer.as_default():
            tf.summary.scalar("discriminator learning rate", self.model.disc_opt.lr, step)
            tf.summary.scalar("generator learning rate", self.model.gen_opt.lr, step)


def get_scheduler(lr, lr_decay):
    if isinstance(lr_decay, str):
        return eval(lr_decay)

    def schedule_lr(step):
        return lr * lr_decay**step
    return schedule_lr

# logarithmic scaler
class Logarithmic:
    def scale(self, x):
        return np.log10(1 + x)
    
    def unscale(self, x):
        return 10 ** x - 1


def train(data_train, data_val, train_step_fn, loss_eval_fn, num_epochs, batch_size,
          train_writer=None, val_writer=None, callbacks=[], features_train=None, features_val=None, first_epoch=0):
    
    if not ((features_train is None) or (features_val is None)):
        assert features_train is not None, 'train: features should be provided for both train and val'
        assert features_val is not None, 'train: features should be provided for both train and val'

    for i_epoch in range(first_epoch, num_epochs):
        print("Working on epoch #{}".format(i_epoch))

        tf.keras.backend.set_learning_phase(1)  # training
        
        shuffle_ids = np.random.permutation(len(data_train))
        losses_train = {}

        for i_sample in trange(0, len(data_train), batch_size):
            batch = data_train[shuffle_ids][i_sample:i_sample + batch_size]
            
            #print ("batch shape = ", batch.shape)
            feature_batch = features_train[shuffle_ids][i_sample:i_sample + batch_size]
            #print ("batch shape = ", feature_batch.shape)
            losses_train_batch = train_step_fn(feature_batch, batch)
            
            for k, l in losses_train_batch.items():
                losses_train[k] = losses_train.get(k, 0) + l.numpy() * len(batch)
                
                
        losses_train = {k : l / len(data_train) for k, l in losses_train.items()}

        tf.keras.backend.set_learning_phase(0)  # testing

        losses_val = {}
        for i_sample in trange(0, len(data_val), batch_size):
            batch = data_val[i_sample:i_sample + batch_size]

            feature_batch = features_val[i_sample:i_sample + batch_size]
            losses_val_batch = {k : l.numpy() for k, l in loss_eval_fn(feature_batch, batch).items()}
            
            for k, l in losses_val_batch.items():
                losses_val[k] = losses_val.get(k, 0) + l * len(batch)
                
        losses_val = {k : l / len(data_val) for k, l in losses_val.items()}

        for f in callbacks:
            f(i_epoch)

        if train_writer is not None:
            with train_writer.as_default():
                for k, l in losses_train.items():
                    tf.summary.scalar(k, l, i_epoch)
        
        if val_writer is not None:
            with val_writer.as_default():
                for k, l in losses_val.items():
                    tf.summary.scalar(k, l, i_epoch)

        print("")
        print("Train losses:", losses_train)
        print("Val losses:", losses_val)
        
def main():
    checkpoint_name = "Second_Try_GPU"
    save_every = 50
    
    continue_training = False
    
    prediction_process = False

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    setup_gpu()
    model_path = _THIS_PATH.joinpath('saved_models',checkpoint_name)
    #model_path = Path('saved_models') / checkpoint_name
    
    if model_path.exists():
        print ("Model directory already exists")
    else: 
        model_path.mkdir(parents=True)
    
    #initializing pad_range and Time_range 
    pad_range=(-3, 5)
    time_range=(-7, 9)
    
    # initializing the model itself
    model = GAN_Model()
    
    
    next_epoch = 0
    if  continue_training or prediction_process:
        next_epoch = load_weights(model, model_path) + 1
    
    #print ("********Generator*******")
    #print (model.generator.summary())
    #print ("********Discriminator*******")
    #print (model.discriminator.summary())
    
    print ("********Model initialized*******")
    
    # reading the csv file and preforming instruction mentioned in the paper 
    
    #we shift the responses by the integer parts of the drift length and pad coordinate along the time and pad row directions, respectively. After having done this, the responses in
    #the whole training set fit onto a matrix of 8 pads by 16 time buckets (data), which constitutes our target space. 
    
    # data shape : (20000,8,16)
    # features shape : (20000,4)
    data, features = read_csv_2d(pad_range=pad_range, time_range=time_range)
    features = features.astype('float32')
    
    
    print ("********Data Read*******")
    
    #The responses span over several orders of magnitude,so we scale them with log10(x + 1) for smoother learning.
    data_scaled = model.scaler.scale(data).astype('float32')
    
    print ("********Data scaled*******")
    
    #spliting the data into train an test data
    Y_train, Y_test, X_train, X_test = train_test_split(data_scaled, features, test_size=0.25, random_state=42)
    
    print ("********Data splitted*******")
    
    
    if not prediction_process:        
        writer_train = tf.summary.create_file_writer(str(_THIS_PATH.joinpath('logs',checkpoint_name, 'train')))
        writer_val = tf.summary.create_file_writer(str(_THIS_PATH.joinpath('logs',checkpoint_name, 'validation')))
    
    
        print ("********Initializing CallBacks*******")
        save_model = SaveModelCallback(
                model=model, path=model_path, save_period=save_every
            )
        
        write_hist_summary = WriteHistSummaryCallback(
                model, sample=(X_test, Y_test),
                save_period=save_every, writer=writer_val
            )
        
        schedule_lr = ScheduleLRCallback(
                model, writer=writer_val,
                func_gen=get_scheduler(model.lr, model.lr_schedule),
                func_disc=get_scheduler(model.lr, model.lr_schedule)
            )
        
        if continue_training:
            schedule_lr(next_epoch -1)
        #print ("Y_train shape = ", Y_train.shape)
        print ("********Data training initialized*******")
        
        
        train(Y_train, Y_test, model.training_step, model.calculate_losses, model.num_epochs, model.batch_size,
                train_writer=writer_train, val_writer=writer_val,
                callbacks=[write_hist_summary, save_model, schedule_lr],
                features_train=X_train, features_val=X_test,first_epoch=next_epoch)
        
        print ("********Data training initialized********")
        
    else:
        epoch = latest_epoch(model_path=model_path)
        pridiction_path = model_path/f"prediction_{epoch:05d}"
        pridiction_path.mkdir()
        
        for part in ['train', 'test']:
            evaluate_model(
                model, path=pridiction_path / part,
                sample=(
                    (X_train, Y_train) if part == 'train'
                    else (X_test, Y_test)
                ),pad_range = pad_range, time_range = time_range,
                gen_sample_name=(None if part == 'train' else 'generated.dat')
            )

    
if __name__ == "__main__" :
    main()