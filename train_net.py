
from Trainer import GANsTrainer
from lib.cifar_dataset_util import CIFARDatahandler
from lib.mnist_dataset_util import MNISTDatahandler
from lib.tfd_dataset_util import TFDDataHandler

datasets_handle = {
	"cifar": CIFARDatahandler,
	"tfd": TFDDataHandler,
	"mnist": MNISTDatahandler
}


if __name__ == "__main__":

	opts = {}
	opts['use_gpu'] = True
	opts['batchSize'] = 20
	opts['numEpochs'] = 40

	opts["expDir"] = "Exp"
	opts["innerTrainSteps"] = 1
	opts["numTrainIte"] = 200000

	opts['discriminatorLearningRate'] = 0.00001
	opts['generatorLearningRate'] = 0.001
	opts['weightDecay'] = 0.005
	opts['momentum'] = 0.09

	opts['continue'] = None
	opts['data_loader'] = datasets_handle["cifar"]

	gans_trainer = GANsTrainer(opts)
	gans_trainer.setup()
	# gans_trainer.process_discriminator_training_round()
	gans_trainer.train_model(opts)
