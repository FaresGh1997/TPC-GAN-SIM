import numpy as np
import pandas as pd
import math
import tensorflow as tf 





"""def read_csv_from_path(path):
    df = pd.read_csv(path)
    input_features = df.iloc[:,-4:]
    return input_features.to_numpy(dtype='float32')
"""


def read_csv_2d(filename=None, pad_range=(40, 50), time_range=(265, 280), strict=True, misc_out=None):
    if filename is None:
        filename = str(_THIS_PATH.joinpath(_VERSION, 'csv', 'digits.csv'))

    df = pd.read_csv(filename)

    def sel(df, col, limits):
        return (df[col] >= limits[0]) & (df[col] < limits[1])

    if 'drift_length' in df.columns:
        df['itime'] -= df['drift_length'].astype(int)

    if 'pad_coordinate' in df.columns:
        df['ipad'] -= df['pad_coordinate'].astype(int)

    selection = sel(df, 'itime', time_range) & sel(df, 'ipad', pad_range)
    g = df[selection].groupby('evtId')
    bad_ids = df[~selection]['evtId'].unique()
    anti_selection = df['evtId'].apply(lambda x: x in bad_ids)
    anti_g = df[anti_selection].groupby('evtId')

    if not selection.all():
        msg = (
            f"WARNING: current selection ignores {(~selection).sum() / len(selection) * 100}% of the data"
            f" ({len(anti_g)} events)!"
        )
        assert not strict, msg
        print(msg)

    def convert_event(event):
        result = np.zeros(dtype=float, shape=(pad_range[1] - pad_range[0], time_range[1] - time_range[0]))

        indices = tuple(event[['ipad', 'itime']].values.T - np.array([[pad_range[0]], [time_range[0]]]))
        result[indices] = event.amp.values

        return result

    data = np.stack(g.apply(convert_event).values)
    anti_data = None
    if not selection.all() and misc_out is not None:
        assert isinstance(misc_out, dict)
        pad_range = [df[anti_selection]["ipad"].min(), df[anti_selection]["ipad"].max() + 1]
        time_range = [df[anti_selection]["itime"].min(), df[anti_selection]["itime"].max() + 1]
        anti_data = np.stack(anti_g.apply(convert_event).values)
        misc_out["anti_data"] = anti_data
        misc_out["bad_ids"] = bad_ids

    if 'crossing_angle' in df.columns:
        features = ['crossing_angle', 'dip_angle']
        if 'drift_length' in df.columns:
            features += ['drift_length']
        if 'pad_coordinate' in df.columns:
            features += ['pad_coordinate']
        if "row" in df.columns:
            features += ["row"]
        if "pT" in df.columns:
            features += ["pT"]
        assert (
            (g[features].std() == 0).all(axis=1) | (g[features].size() == 1)
        ).all(), 'Varying features within same events...'
        return data, g[features].mean().values

    return data
@tf.function
def preprocessing_func_as_tensor(input_tensors):
    bin_fractions = input_tensors[:, 2:4] % 1
    input_tensors = (input_tensors[:, :3] - tf.constant([[0.0, 0.0, 162.5]])) / tf.constant([[20.0, 60.0, 127.5]])
    return tf.concat([input_tensors, bin_fractions], axis=-1)

def custom_activation_V2 (x, shift = 0.01,val = np.log10(2),val0 = np.log10(2) / 10):
    return tf.where(x> shift,
                    val +x -shift,
                    val0 + tf.keras.activations.elu(
                        x,alpha = (val0 *shift/ (val -val0))
                                                    ) * (val -val0)/shift)
    
def create_generator_structure():
    
    generator = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation='elu', kernel_initializer= 'glorot_uniform'),
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
    input_img = tf.keras.Input(shape= (8,16))
    features = tf.keras.Input(shape= (5,))    
    
    # reshaping 
    
    # remarks : If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
    
    
    img = tf.reshape(input_img, (-1,8,16,1))
    
    
    features_Tiled = tf.tile((tf.reshape(features, (-1,1,1,5))), (1,8,16,1))
    
    # remarks : Negative axis are interpreted as counting from the end of the rank, i.e., axis + rank(values)-th dimension.
    # remarks : The rank of a tensor is the number of indices required to uniquely select each element of the tensor.

    input_img = tf.concat([img,features_Tiled],-1) 
    
    
    
    
    
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
    
    head_input  = tf.keras.layers.Concatenate()([features,discriminator_main(input_img)])
    
    head_layers  = [
        tf.keras.layers.Dense(128,activation='elu'),
        tf.keras.layers.Dropout(0.02),
        
        
        tf.keras.layers.Dense(1,activation=None)
    ]
    
    discriminator_head = tf.keras.Sequential(
        head_layers,
        name = 'discriminator_head'
    )
    
    inputs = [features, input_img]
    outputs = discriminator_head(head_input)
    
    discriminator = tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')

    return discriminator
    
    
def gen_loss (fake,real):
    #return tf.reduce_mean (real) - tf.reduce_mean(fake)
    return tf.reduce_mean (real - fake)


def disc_loss (fake, real):
    #return tf.reduce_mean (real) - tf.reduce_mean(fake)
    return tf.reduce_mean (fake - real)
   

class GAN_Model:
    def __init__(self) -> None:
        self.preprocess_function = preprocessing_func_as_tensor
        
        self.latent_dim = 32
        self.batch_size = 32
        
        # We train both generator and discriminator using RMSprop optimizer with learning rates starting at 0.0001
        
        self.lr = 1.e-4
        self.lr_schedule = 0.999
        self.disc_opt = tf.keras.optimizers.RMSprop(learning_rate= self.lr)
        self.gen_opt = tf.keras.optimizers.RMSprop (learning_rate= self.lr)
        
        # gradient penalty value
        self.lambda_value = 10
        
        #We make 8 discriminator update steps per single generator step.
        self.num_update_step = 8
        
        
        self.latent_dim = 32
        self.batch_size = 32
        
        
        self.num_epochs = 10000
        
        self.generator = create_generator_structure()
        self.discriminator = create_discriminator_structure()
        
        
        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)
        
        
        self.generator.compile(optimizer=self.gen_opt, loss='mean_squared_error')
        self.discriminator.compile(optimizer=self.disc_opt, loss='mean_squared_error')
            
    @tf.function
    def make_fake (self,features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size,self.latent_dim),dtype='float32')
        return tf.concat([self.preprocess_function(features), latent_input],axis= -1)
    
    def gradient_penalty(self, features, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1])
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator([self.preprocess_function(features), interpolates])
        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0) ** 2)
    
    @tf.function
    def calculate_losses(self, feature_batch, target_batch):
        fake = self.make_fake(feature_batch)
        d_real = self.discriminator([self.preprocess_function(feature_batch), target_batch])
        d_fake = self.discriminator([self.preprocess_function(feature_batch), fake])
        
        # calculating loss
        d_loss = disc_loss(d_real, d_fake)
        # adding Gradient Penalty 
        d_loss = d_loss + self.gradient_penalty(feature_batch, target_batch, fake) * self.gp_lambda
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



class SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.path = path
        self.save_period = save_period

    def __call__(self, step):
        if step % self.save_period == 0:
            print(f'Saving model on step {step} to {self.path}')
            self.model.generator.save(str(self.path.joinpath("generator_{:05d}.h5".format(step))))
            self.model.discriminator.save(str(self.path.joinpath("discriminator_{:05d}.h5".format(step))))


def main():
    path = 'C:\TPC_GAN_SIM\TPC-GAN-SIM\Data\digits.csv'
    data_raw = read_csv_from_path(path=path)
    model = GAN_Model()
    print (model.make_fake(data_raw))
    
    
if __name__ == "__main__" :
    main()