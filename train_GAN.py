from pipeline import TrainGANPipeline
from net_definition import GAN

gan = GAN()
pipeline = TrainGANPipeline(gan)
pipeline.train_Discriminator()
pipeline.train_GAN()
