import keras
from keras import layers as L

from embedders import embedding_datasets
from embedders.embedder import Embedder


class KerasEmbedder(Embedder):
    def __init__(self, model_path: str = None, dataset: embedding_datasets.CorpusEmbeddingDataset = None):
        super().__init__(model_path, dataset)
        self.__auto_encoder = None
        self._val_loss = None

    @property
    def _auto_encoder(self):
        return self.__auto_encoder

    @_auto_encoder.setter
    def _auto_encoder(self, auto_encoder):
        self.__auto_encoder = auto_encoder
        self._embedder_model = keras.models.Model(self._auto_encoder.input, self._auto_encoder.layers[1].output)
        self._vocab_embeddings = self.embed(one_hots=self._dataset.encoded_vocab)

    def fit(self, dataset=None, epochs=100, embedding_dims=100, softmax=True, verbose=True):
        """
        builds & trains a new embedding network using training data (from an encoder) and specified epochs
        :return: embedding network, final_validation_loss
        """
        if dataset: self._dataset = dataset
        in_data, out_data = self._dataset.get_training_data()

        last_activation = 'softmax' if softmax else None

        in_size = in_data.shape[1]
        in_layer = L.Input((in_size,))
        layer = L.Dense(embedding_dims, use_bias=False)(in_layer)
        layer = L.Dropout(rate=0.4)(layer)  # we were overfitting
        output_layer = L.Dense(in_size, use_bias=False, activation=last_activation)(layer)

        # auto-encoder used as an intermediate model to get embedder
        auto_encoder = keras.models.Model(in_layer, output_layer)
        auto_encoder.compile(keras.optimizers.Adam(), 'categorical_crossentropy')

        training_history = auto_encoder.fit(in_data, out_data, epochs=epochs,
                                                  validation_split=0.2, verbose=verbose)
        self._val_loss = training_history.history['val_loss'][-1]
        self._auto_encoder = auto_encoder
        # we assign here so that vocabs are recomputed after training

    def _embed(self, words=None, one_hots=None):
        if words is not None: one_hots = self._dataset.encode(words)

        embeddings = self.embedder_model.predict(one_hots)
        return embeddings

    def load(self, model_path):
        self._auto_encoder = keras.models.load_model(model_path)

    def save(self, model_path):
        self._auto_encoder.save(model_path)

    @property
    def _embedding_weights(self):
        return self._embedding_layer.get_weights()[0].T, 0

    @property
    def _embedding_layer(self):
        return self._embedder_model.layers[1]

    @property
    def val_loss(self):
        if self._val_loss is not None:
            return self._val_loss
        else:
            raise RuntimeError("model is pretrained val_loss is unavailable")
                  #self._embedder_model.evaluate(*self._dataset.get_training_data())
