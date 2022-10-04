# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-09-29T16:04:45.737699Z","iopub.execute_input":"2022-09-29T16:04:45.738195Z","iopub.status.idle":"2022-09-29T16:04:50.733131Z","shell.execute_reply.started":"2022-09-29T16:04:45.738070Z","shell.execute_reply":"2022-09-29T16:04:50.731868Z"}}
import os
# Do not print warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from glob import glob
import shutil
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

AUTOTUNE = tf.data.AUTOTUNE

# %% [markdown]
# We unzip our training dataset.

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:04:50.740171Z","iopub.execute_input":"2022-09-29T16:04:50.740930Z","iopub.status.idle":"2022-09-29T16:05:15.278521Z","shell.execute_reply.started":"2022-09-29T16:04:50.740879Z","shell.execute_reply":"2022-09-29T16:05:15.277405Z"}}

# %% [markdown]
# Print training and test dataset sizes.

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:05:15.280136Z","iopub.execute_input":"2022-09-29T16:05:15.280539Z","iopub.status.idle":"2022-09-29T16:05:15.448424Z","shell.execute_reply.started":"2022-09-29T16:05:15.280502Z","shell.execute_reply":"2022-09-29T16:05:15.447353Z"}}
print(f"Number of dogs: {len(glob('./train/dog/*'))}\nNumber of cats: {len(glob('./train/cat/*'))}")

# %% [markdown]
# Ahora creamos las listas de directorios para las imágenes de perros ($X$) y las imágenes de gatos ($Y$).

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:05:15.451270Z","iopub.execute_input":"2022-09-29T16:05:15.451925Z","iopub.status.idle":"2022-09-29T16:05:15.556295Z","shell.execute_reply.started":"2022-09-29T16:05:15.451882Z","shell.execute_reply":"2022-09-29T16:05:15.555218Z"}}
# Ahora definimos unas cuantas variables globales.

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:05:15.557962Z","iopub.execute_input":"2022-09-29T16:05:15.558362Z","iopub.status.idle":"2022-09-29T16:05:15.564531Z","shell.execute_reply.started":"2022-09-29T16:05:15.558322Z","shell.execute_reply":"2022-09-29T16:05:15.562913Z"}}
BUFFER_SIZE = 12500
BATCH_SIZE = 32
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

# %% [markdown]
# Definimos las siguientes utilidades para la aumentación de datos.

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:05:15.566860Z","iopub.execute_input":"2022-09-29T16:05:15.567675Z","iopub.status.idle":"2022-09-29T16:05:15.576798Z","shell.execute_reply.started":"2022-09-29T16:05:15.567635Z","shell.execute_reply":"2022-09-29T16:05:15.575456Z"}}
# randomly crop an IMG_HEIGHT x IMG_WIDTH patch

norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)
# normalizing the images to [-1, 1]

# %% [markdown] {"execution":{"iopub.status.busy":"2022-09-27T22:43:46.164898Z","iopub.execute_input":"2022-09-27T22:43:46.165342Z","iopub.status.idle":"2022-09-27T22:43:46.172424Z","shell.execute_reply.started":"2022-09-27T22:43:46.165303Z","shell.execute_reply":"2022-09-27T22:43:46.171018Z"}}
# Definimos la función que utilizará el data loader para cargar las imágenes

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:05:50.880875Z","iopub.execute_input":"2022-09-29T16:05:50.881628Z","iopub.status.idle":"2022-09-29T16:05:54.227898Z","shell.execute_reply.started":"2022-09-29T16:05:50.881585Z","shell.execute_reply":"2022-09-29T16:05:54.226881Z"}}
train_X = tf.keras.utils.image_dataset_from_directory(directory='./train/dog/', labels=None, seed=123, batch_size=BATCH_SIZE).take(200)
train_X = train_X.map(lambda image: norm_layer(image), num_parallel_calls=AUTOTUNE)

train_Y = tf.keras.utils.image_dataset_from_directory(directory='./train/cat/', labels=None, seed=123, batch_size=BATCH_SIZE).take(200)
train_Y = train_Y.map(lambda image: norm_layer(image), num_parallel_calls=AUTOTUNE)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:05:59.617164Z","iopub.execute_input":"2022-09-29T16:05:59.617562Z","iopub.status.idle":"2022-09-29T16:06:06.067741Z","shell.execute_reply.started":"2022-09-29T16:05:59.617522Z","shell.execute_reply":"2022-09-29T16:06:06.066638Z"}}
sample_dog = next(iter(train_X))
sample_cat = next(iter(train_Y))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:06:13.705711Z","iopub.execute_input":"2022-09-29T16:06:13.706115Z","iopub.status.idle":"2022-09-29T16:06:14.099778Z","shell.execute_reply.started":"2022-09-29T16:06:13.706059Z","shell.execute_reply":"2022-09-29T16:06:14.097760Z"}}
#plt.figure(figsize=(12,6))
#plt.subplot(121)
#plt.title('Dog')
#plt.imshow(sample_dog[0] * 0.5 + 0.5)
#plt.axis('off')

#plt.subplot(122)
#plt.title('Dog with random jitter')
#plt.imshow(random_jitter(sample_dog[0]) * 0.5 + 0.5)
#plt.axis('off')

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:06:14.646984Z","iopub.execute_input":"2022-09-29T16:06:14.647781Z","iopub.status.idle":"2022-09-29T16:06:14.998965Z","shell.execute_reply.started":"2022-09-29T16:06:14.647719Z","shell.execute_reply":"2022-09-29T16:06:14.997963Z"}}
#plt.figure(figsize=(12,6))
#plt.subplot(121)
#plt.title('Cat')
#plt.imshow(sample_cat[0] * 0.5 + 0.5)
#plt.axis('off')

#plt.subplot(122)
#plt.title('Cat with random jitter')
#plt.imshow(random_jitter(sample_cat[0]) * 0.5 + 0.5)
#plt.axis('off')

# %% [markdown]
# ## Definición del modelo Generador
# Comenzamos con la definición de los bloques codificadores y decodificadores del modelo tipo UNet. Por último, verificamos que las dimensiones de salida.

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:06:27.520498Z","iopub.execute_input":"2022-09-29T16:06:27.520908Z","iopub.status.idle":"2022-09-29T16:06:34.888197Z","shell.execute_reply.started":"2022-09-29T16:06:27.520873Z","shell.execute_reply":"2022-09-29T16:06:34.886273Z"}}
# Instance Normalization Layer
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     shape=input_shape[-1:],
                                     initializer=tf.random_normal_initializer(1., 0.02),
                                     trainable=True)

        self.offset = self.add_weight(name='offset',
                                      shape=input_shape[-1:], 
                                      initializer='zeros', 
                                      trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

# Down-sampling block
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=size,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    
    if apply_batchnorm:
        result.add(InstanceNormalization()) # Perform instance normalization

    result.add(tf.keras.layers.LeakyReLU())

    return result

# Downsampling code verification
down_model = downsample(3,4)
down_result = down_model(tf.expand_dims(sample_dog[0], 0))

# Up-sampling block
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters=filters,
                                               kernel_size=size,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))

    result.add(InstanceNormalization()) # Perform instance normalization

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

# Upsampling verification
up_model = upsample(3, 4)
up_result = up_model(down_result)
print (f'Output dimensions{up_result.shape}')

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T16:06:34.890289Z","iopub.execute_input":"2022-09-29T16:06:34.890679Z","iopub.status.idle":"2022-09-29T16:06:34.903371Z","shell.execute_reply.started":"2022-09-29T16:06:34.890638Z","shell.execute_reply":"2022-09-29T16:06:34.902143Z"}}
# UNet Generator
def Generator():
    # Layers that compose the net
    x_input = tf.keras.layers.Input(shape=(256, 256, 3))
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,
                                           4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (batch_size, 256, 256, 1)

    # Processing pipeline
    x = x_input
    
    # Encoder
    skips=[]
    for down in down_stack:
        x = down(x)
        skips.append(x) # Output for each downsampling is added to a list

    skips = reversed(skips[:-1])
    
    #Decoder
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)

    return tf.keras.Model(inputs=x_input, outputs=x)

# %% [markdown]
# Probamos las salidas de los generadores $G$ y $F$, con pesos inicializados aleatoriamente.

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T01:33:24.502857Z","iopub.execute_input":"2022-09-29T01:33:24.503886Z","iopub.status.idle":"2022-09-29T01:33:26.687772Z","shell.execute_reply.started":"2022-09-29T01:33:24.503845Z","shell.execute_reply":"2022-09-29T01:33:26.686195Z"}}
# Test generators
generator_G = Generator()
gen_output = generator_G(sample_dog[0][tf.newaxis, ...], training=False)
generator_F = Generator()
gen_output2 = generator_F(sample_cat[0][tf.newaxis, ...], training=False)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title('Dog to cat, $G(X)$')
plt.imshow(gen_output[0, ...]*50, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.title('Cat to dog, $F(Y)$')
plt.imshow(gen_output2[0, ...]*50, cmap='gray')
plt.axis('off')
plt.savefig('out/Test_Random_Generators.png')
plt.clf()
#plt.show()

# %% [markdown]
# Ahora procedemos a definir nuestros discriminadores.

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T01:33:26.689391Z","iopub.execute_input":"2022-09-29T01:33:26.690602Z","iopub.status.idle":"2022-09-29T01:33:26.701660Z","shell.execute_reply.started":"2022-09-29T01:33:26.690552Z","shell.execute_reply":"2022-09-29T01:33:26.700645Z"}}
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    x = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    instnorm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(instnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=x, outputs=last)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T04:06:14.783844Z","iopub.execute_input":"2022-09-29T04:06:14.784257Z","iopub.status.idle":"2022-09-29T04:06:15.205479Z","shell.execute_reply.started":"2022-09-29T04:06:14.784200Z","shell.execute_reply":"2022-09-29T04:06:15.204280Z"}}
# Test discriminators
discriminator_X = Discriminator()
discriminator_Y = Discriminator()

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.title('Is a real dog?')
plt.axis('off')
plt.imshow(discriminator_X(sample_cat)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real cat?')
plt.axis('off')
plt.imshow(discriminator_Y(sample_dog)[0, ..., -1], cmap='RdBu_r')

plt.savefig('out/Test_Random_Discriminators.png')
plt.clf()
#plt.show()



# %% [markdown]
# Next, we define our loss functions.

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T01:34:26.539967Z","iopub.execute_input":"2022-09-29T01:34:26.540563Z","iopub.status.idle":"2022-09-29T01:34:26.551148Z","shell.execute_reply.started":"2022-09-29T01:34:26.540515Z","shell.execute_reply":"2022-09-29T01:34:26.549980Z"}}
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss(discriminated):
    return loss_obj(tf.ones_like(discriminated), discriminated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    
    return LAMBDA * 0.5 * loss

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T01:33:27.206253Z","iopub.execute_input":"2022-09-29T01:33:27.207341Z","iopub.status.idle":"2022-09-29T01:33:27.218857Z","shell.execute_reply.started":"2022-09-29T01:33:27.207278Z","shell.execute_reply":"2022-09-29T01:33:27.217699Z"}}
generator_G_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_F_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_X_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_Y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T01:33:27.221369Z","iopub.execute_input":"2022-09-29T01:33:27.221998Z","iopub.status.idle":"2022-09-29T01:33:27.234186Z","shell.execute_reply.started":"2022-09-29T01:33:27.221957Z","shell.execute_reply":"2022-09-29T01:33:27.233383Z"}}
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_G=generator_G,
                           generator_F=generator_F,
                           discriminator_X=discriminator_X,
                           discriminator_Y=discriminator_Y,
                           generator_G_optimizer=generator_G_optimizer,
                           generator_F_optimizer=generator_F_optimizer,
                           discriminator_X_optimizer=discriminator_X_optimizer,
                           discriminator_Y_optimizer=discriminator_Y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

# %% [markdown]
# ## Training

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T02:24:03.538818Z","iopub.execute_input":"2022-09-29T02:24:03.539271Z","iopub.status.idle":"2022-09-29T02:24:03.546740Z","shell.execute_reply.started":"2022-09-29T02:24:03.539225Z","shell.execute_reply":"2022-09-29T02:24:03.545487Z"}}
def generate_images(model, test_input, fname):
    prediction = model(test_input)
    
    plt.figure(figsize=(12,12))
    
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(fname)
    plt.clf()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T01:36:30.495347Z","iopub.execute_input":"2022-09-29T01:36:30.495851Z","iopub.status.idle":"2022-09-29T01:36:30.519459Z","shell.execute_reply.started":"2022-09-29T01:36:30.495808Z","shell.execute_reply":"2022-09-29T01:36:30.517796Z"}}
@tf.function
def train_step(real_X, real_Y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X
        
        fake_Y = generator_G(real_X, training=True)
        cycled_X = generator_F(fake_Y, training=True)
        
        fake_X = generator_F(real_Y, training=True)
        cycled_Y = generator_G(fake_X, training=True)
        
        # same_X and same_Y are used for identity loss
        same_X = generator_F(real_X, training=True)
        same_Y = generator_G(real_Y, training=True)
        
        disc_real_X = discriminator_X(real_X, training=True)
        disc_real_Y = discriminator_Y(real_Y, training=True)
        
        disc_fake_X = discriminator_X(fake_X, training=True)
        disc_fake_Y = discriminator_Y(fake_Y, training=True)
        
        # calculate the loss
        gen_G_loss = generator_loss(disc_fake_Y)
        gen_F_loss = generator_loss(disc_fake_X)
        
        total_cycle_loss = calc_cycle_loss(real_X, cycled_X) + calc_cycle_loss(real_Y, cycled_Y)
        
        # total generator loss = adversarial loss + cycle loss
        total_gen_G_loss = gen_G_loss + total_cycle_loss + identity_loss(real_Y, same_Y)
        total_gen_F_loss = gen_F_loss + total_cycle_loss + identity_loss(real_X, same_X)
        
        disc_X_loss = discriminator_loss(disc_real_X, disc_fake_X)
        disc_Y_loss = discriminator_loss(disc_real_Y, disc_fake_Y)
        
    # Calculate the gradients for generator and discriminator
    generator_G_gradients = tape.gradient(total_gen_G_loss, 
                                              generator_G.trainable_variables)
    generator_F_gradients = tape.gradient(total_gen_F_loss,
                                              generator_F.trainable_variables)
        
    discriminator_X_gradients = tape.gradient(disc_X_loss, 
                                                  discriminator_X.trainable_variables)
    discriminator_Y_gradients = tape.gradient(disc_Y_loss, 
                                                  discriminator_Y.trainable_variables)
        
    # Apply the gradients to the optimizer
    generator_G_optimizer.apply_gradients(zip(generator_G_gradients, 
                                                  generator_G.trainable_variables))

    generator_F_optimizer.apply_gradients(zip(generator_F_gradients, 
                                                  generator_F.trainable_variables))

    discriminator_X_optimizer.apply_gradients(zip(discriminator_X_gradients,
                                                      discriminator_X.trainable_variables))

    discriminator_Y_optimizer.apply_gradients(zip(discriminator_Y_gradients,
                                                      discriminator_Y.trainable_variables))
        
        
        

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T02:24:50.758717Z","iopub.execute_input":"2022-09-29T02:24:50.759080Z","iopub.status.idle":"2022-09-29T04:03:32.669000Z","shell.execute_reply.started":"2022-09-29T02:24:50.759048Z","shell.execute_reply":"2022-09-29T04:03:32.667435Z"}}
EPOCHS = 50 
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_X, image_Y in tf.data.Dataset.zip((train_X, train_Y)):
        train_step(image_X, image_Y)
        if n % 10 == 0:
            print ('.', end='')
        n += 1
        
    # Using a consistent image (sample_dog) so that the progress of the model
    #clear_output(wait=True)
    if (epoch + 1) % 10 == 0:
        generate_images(generator_G, sample_dog, f'out/train/epoch_{epoch+1}_dog.png')
        generate_images(generator_F, sample_cat, f'out/train/epoch_{epoch+1}_cat.png')
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                             ckpt_save_path))
        
    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T04:03:46.182499Z","iopub.execute_input":"2022-09-29T04:03:46.182859Z","iopub.status.idle":"2022-09-29T04:03:51.309411Z","shell.execute_reply.started":"2022-09-29T04:03:46.182828Z","shell.execute_reply":"2022-09-29T04:03:51.308338Z"}}
# Run the trained model on the test dataset
inp = next(iter(train_X))
for k in range(8):
    generate_images(generator_G, inp[k][tf.newaxis, ...], f'out/test/test_{k+1}_dog.png')

inp = next(iter(train_Y))
for k in range(8):
    generate_images(generator_F, inp[k][tf.newaxis, ...], f'out/test/test_{k+1}_cat.png')
