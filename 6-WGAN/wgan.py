from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers

class WGAN():
    def __init__(self, generator, critic):

        self.name = 'wgan'
        self.generator = generator
        self.critic = critic
        
        self.z_dim = generator.input_shape[1:]
        
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
    
    def build_model(self
                   , optimizer
                   , generator_lr
                   , critic_lr
                   ):
        
        def get_opti(lr):
            if optimizer == 'adam':
                opti = optimizers.Adam(lr=lr, beta_1=0.5)
            elif optimizer == 'rmsprop':
                opti = optimizers.RMSprop(lr=lr)
            else:
                opti = optimizers.Adam(lr=lr)
            return opti
        
        def wasserstein(y_true, y_pred):
            return - K.mean(y_true * y_pred)
        
        def set_trainable(m, val):
            m.trainable = val
            for l in m.layers:
                l.trainable = val
        
        ### COMPILE critic
        self.critic.compile( optimizer = get_opti(critic_lr)
                             , loss = wasserstein
                           )
        
        ### COMPILE THE FULL GAN
        set_trainable(self.critic, False)

        model_input = layers.Input(shape=self.z_dim, name='model_input')
        model_output = self.critic(self.generator(model_input))
        self.model = models.Model(model_input, model_output)
        
        self.model.compile(
            optimizer=get_opti(generator_lr)
            , loss=wasserstein
            )

        set_trainable(self.critic, True)

