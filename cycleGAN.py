# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-09-29T16:04:45.737699Z","iopub.execute_input":"2022-09-29T16:04:45.738195Z","iopub.status.idle":"2022-09-29T16:04:50.733131Z","shell.execute_reply.started":"2022-09-29T16:04:45.738070Z","shell.execute_reply":"2022-09-29T16:04:50.731868Z"}}
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import numpy as np
import shutil
import time
import os
# Do not print warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


AUTOTUNE = tf.data.AUTOTUNE

print(
    f"Number of dogs: {len(glob('./train/dog/*'))}\nNumber of cats: {len(glob('./train/cat/*'))}")


BUFFER_SIZE = 12500
R_LOSS_FACTOR = 100000  # 10000
LATENT_DIM = 150
BATCH_SIZE = 32
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3


norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
    scale=1./127.5, offset=-1)
# normalizing the images to [-1, 1]

train_X = tf.keras.utils.image_dataset_from_directory(
    directory='./train/dog/', labels=None, seed=123, batch_size=BATCH_SIZE).take(200)
train_X = train_X.map(lambda image: norm_layer(image),
                      num_parallel_calls=AUTOTUNE)

train_Y = tf.keras.utils.image_dataset_from_directory(
    directory='./train/cat/', labels=None, seed=123, batch_size=BATCH_SIZE).take(200)
train_Y = train_Y.map(lambda image: norm_layer(image),
                      num_parallel_calls=AUTOTUNE)

sample_dog = next(iter(train_X))
sample_cat = next(iter(train_Y))


class VAE(keras.Model):
    def __init__(self, r_loss_factor=1, summary=False, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.r_loss_factor = r_loss_factor

        # Architecture
        self.input_dim = INPUT_DIM
        self.latent_dim = LATENT_DIM
        self.encoder_conv_filters = [64, 64, 64, 64]
        self.encoder_conv_kernel_size = [3, 3, 3, 3]
        self.encoder_conv_strides = [2, 2, 2, 2]
        self.n_layers_encoder = len(self.encoder_conv_filters)

        self.decoder_conv_t_filters = [64, 64, 64, 3]
        self.decoder_conv_t_kernel_size = [3, 3, 3, 3]
        self.decoder_conv_t_strides = [2, 2, 2, 2]
        self.n_layers_decoder = len(self.decoder_conv_t_filters)

        self.use_batch_norm = True
        self.use_dropout = True

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.mae = tf.keras.losses.MeanAbsoluteError()

        # Encoder
        self.encoder_model = Encoder(input_dim=self.input_dim,
                                     output_dim=self.latent_dim,
                                     encoder_conv_filters=self.encoder_conv_filters,
                                     encoder_conv_kernel_size=self.encoder_conv_kernel_size,
                                     encoder_conv_strides=self.encoder_conv_strides,
                                     use_batch_norm=self.use_batch_norm,
                                     use_dropout=self.use_dropout)
        self.encoder_conv_size = self.encoder_model.last_conv_size
        if summary:
            self.encoder_model.summary()

        # Sampler
        self.sampler_model = Sampler(latent_dim=self.latent_dim)
        if summary:
            self.sampler_model.summary()

        # Decoder
        self.decoder_model = Decoder(input_dim=self.latent_dim,
                                     input_conv_dim=self.encoder_conv_size,
                                     decoder_conv_t_filters=self.decoder_conv_t_filters,
                                     decoder_conv_t_kernel_size=self.decoder_conv_t_kernel_size,
                                     decoder_conv_t_strides=self.decoder_conv_t_strides,
                                     use_batch_norm=self.use_batch_norm,
                                     use_dropout=self.use_dropout)
        if summary:
            self.decoder_model.summary()

        self.built = True

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker, ]

    @tf.function
    def train_step(self, data):
        '''
        '''
        with tf.GradientTape() as tape:

            # predict
            x = self.encoder_model(data)
            z, z_mean, z_log_var = self.sampler_model(x)
            pred = self.decoder_model(z)

            # loss
            r_loss = self.r_loss_factor * self.mae(data, pred)
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = r_loss + kl_loss

        # gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        # train step
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # compute progress
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss":                self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss":             self.kl_loss_tracker.result(), }

    @tf.function
    def generate(self, z_sample):
        '''
        We use the sample of the N(0,I) directly as  
        input of the deterministic generator. 
        '''
        return self.decoder_model(z_sample)

    @tf.function
    def codify(self, images):
        '''
        For an input image we obtain its particular distribution:
        its mean, its variance (unvertaintly) and a sample z of such distribution.
        '''
        x = self.encoder_model.predict(images)
        z, z_mean, z_log_var = self.sampler_model(x)
        return z, z_mean, z_log_var

    # implement the call method
    @tf.function
    def call(self, inputs, training=False):
        '''
        '''
        tmp1, tmp2 = self.encoder_model.use_Dropout, self.decoder_model.use_Dropout
        if not training:
            self.encoder_model.use_Dropout, self.decoder_model.use_Dropout = False, False

        x = self.encoder_model(inputs)
        z, z_mean, z_log_var = self.sampler_model(x)
        pred = self.decoder_model(z)

        self.encoder_model.use_Dropout, self.decoder_model.use_Dropout = tmp1, tmp2
        return pred


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     shape=input_shape[-1:],
                                     initializer=tf.random_normal_initializer(
                                         1., 0.02),
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
        result.add(InstanceNormalization())  # Perform instance normalization

    result.add(tf.keras.layers.LeakyReLU())

    return result


# Downsampling code verification
down_model = downsample(3, 4)
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

    result.add(InstanceNormalization())  # Perform instance normalization

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# Upsampling verification
up_model = upsample(3, 4)
up_result = up_model(down_result)
print(f'Output dimensions{up_result.shape}')

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
                                           activation='tanh')  # (batch_size, 256, 256, 1)

    # Processing pipeline
    x = x_input

    # Encoder
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)  # Output for each downsampling is added to a list

    skips = reversed(skips[:-1])

    # Decoder
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=x_input, outputs=x)


# Test generators
generator_G = Generator()
gen_output = generator_G(sample_dog[0][tf.newaxis, ...], training=False)
generator_F = Generator()
gen_output2 = generator_F(sample_cat[0][tf.newaxis, ...], training=False)

plt.figure(figsize=(12, 6))
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
# plt.show()


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

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(
        leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=x, outputs=last)


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
# plt.show()
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


vae_x = VAE(R_LOSS_FACTOR)
vae_y = VAE(R_LOSS_FACTOR)
generator_G_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_F_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_X_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_Y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(
    vae_x=vae_x,
    vae_y=vae_y,
    generator_G=generator_G,
    generator_F=generator_F,
    discriminator_X=discriminator_X,
    discriminator_Y=discriminator_Y,
    generator_G_optimizer=generator_G_optimizer,
    generator_F_optimizer=generator_F_optimizer,
    discriminator_X_optimizer=discriminator_X_optimizer,
    discriminator_Y_optimizer=discriminator_Y_optimizer
)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# ## Training


def generate_images(model, test_input, fname):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

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


@tf.function
def train_step(real_X, real_Y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X
        x = vae_x.encoder_model(real_X)
        zx, zx_mean, zx_log_var = vae_x.sampler_model(x)
        pred_x = vae_x.decoder_model(zx)

        # loss
        rx_loss = vae_x.r_loss_factor * vae_x.mae(real_X,
                                                  pred_x)
        klx_loss = -0.5 * (1 + zx_log_var -
                           tf.square(zx_mean) - tf.exp(zx_log_var))
        klx_loss = tf.reduce_mean(tf.reduce_sum(klx_loss, axis=1))
        total_x_loss = rx_loss + klx_loss

        y = vae_y.encoder_model(real_Y)
        zy, zy_mean, zy_log_var = vae_y.sampler_model(y)
        pred_y = vae_y.decoder_model(zy)

        # loss
        ry_loss = vae_y.r_loss_factor * vae_y.mae(real_Y,
                                                  pred_y)
        kly_loss = -0.5 * (1 + zy_log_var -
                           tf.square(zy_mean) - tf.exp(zy_log_var))
        kly_loss = tf.reduce_mean(tf.reduce_sum(kly_loss, axis=1))
        total_y_loss = ry_loss + kly_loss

        fake_Y = generator_G(pred_x, training=True)
        cycled_X = generator_F(fake_Y, training=True)

        fake_X = generator_F(pred_y, training=True)
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

        total_cycle_loss = calc_cycle_loss(
            real_X, cycled_X) + calc_cycle_loss(real_Y, cycled_Y)

        # total generator loss = adversarial loss + cycle loss
        total_gen_G_loss = gen_G_loss + \
            total_cycle_loss + identity_loss(real_Y, same_Y)
        total_gen_F_loss = gen_F_loss + \
            total_cycle_loss + identity_loss(real_X, same_X)

        disc_X_loss = discriminator_loss(disc_real_X, disc_fake_X)
        disc_Y_loss = discriminator_loss(disc_real_Y, disc_fake_Y)

    vae_x_grads = tape.gradient(total_x_loss,
                                vae_x.trainable_weights)
    vae_y_grads = tape.gradient(total_y_loss,
                                vae_y.trainable_weights)
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
    vae_x.optimizer.apply_gradients(zip(vae_x_grads,
                                        vae_x.trainable_weights))
    vae_y.optimizer.apply_gradients(zip(vae_y_grads,
                                        vae_y.trainable_weights))
    generator_G_optimizer.apply_gradients(zip(generator_G_gradients,
                                              generator_G.trainable_variables))

    generator_F_optimizer.apply_gradients(zip(generator_F_gradients,
                                              generator_F.trainable_variables))

    discriminator_X_optimizer.apply_gradients(zip(discriminator_X_gradients,
                                                  discriminator_X.trainable_variables))

    discriminator_Y_optimizer.apply_gradients(zip(discriminator_Y_gradients,
                                                  discriminator_Y.trainable_variables))


EPOCHS = 50
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_X, image_Y in tf.data.Dataset.zip((train_X, train_Y)):
        train_step(image_X, image_Y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

    # Using a consistent image (sample_dog) so that the progress of the model
    # clear_output(wait=True)
    if (epoch + 1) % 10 == 0:
        generate_images(generator_G, sample_dog,
                        f'out/train/epoch_{epoch+1}_dog.png')
        generate_images(generator_F, sample_cat,
                        f'out/train/epoch_{epoch+1}_cat.png')
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time()-start))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-29T04:03:46.182499Z","iopub.execute_input":"2022-09-29T04:03:46.182859Z","iopub.status.idle":"2022-09-29T04:03:51.309411Z","shell.execute_reply.started":"2022-09-29T04:03:46.182828Z","shell.execute_reply":"2022-09-29T04:03:51.308338Z"}}
# Run the trained model on the test dataset
inp = next(iter(train_X))
for k in range(8):
    generate_images(
        generator_G, inp[k][tf.newaxis, ...], f'out/test/test_{k+1}_dog.png')

inp = next(iter(train_Y))
for k in range(8):
    generate_images(
        generator_F, inp[k][tf.newaxis, ...], f'out/test/test_{k+1}_cat.png')
