from ctgan import CTGAN
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

class TVAESynthesizer:
    """TVAE synthetic data generator."""
    def __init__(self, epochs=50, batch_size=256):
        """
        Initialize TVAE model.
        Args:
            epochs: Number of training iterations.
            batch_size: Number of samples per batch.
        """
        self.model = Plugins().get(
            "tvae",
            n_iter=epochs,
            batch_size=batch_size,
            decoder_n_layers_hidden=3,
            encoder_n_layers_hidden=3
        )

    def train(self, data, categorical_cols):
        """Train the TVAE model."""
        dataloader = GenericDataLoader(
            data,
            target_column=data.columns[-1],
            sensitive_columns=[],
            random_state=0
        )
        self.model.fit(dataloader)

    def generate(self, num_samples):
        """Generate synthetic data."""
        return self.model.generate(count=num_samples).dataframe()

class NFLOWSynthesizer:
    """NFLOW synthetic data generator."""
    def __init__(self, epochs=10, batch_size=256):
        """
        Initialize NFLOW model.
        Args:
            epochs: Number of training iterations.
            batch_size: Number of samples per batch.
        """
        self.model = Plugins().get("nflow", n_iter=epochs, batch_size=batch_size)

    def train(self, data, categorical_cols):
        """Train the NFLOW model."""
        dataloader = GenericDataLoader(
            data,
            target_column=data.columns[-1],
            sensitive_columns=[],
            random_state=0,
            categorical_columns=categorical_cols
        )
        self.model.fit(dataloader)

    def generate(self, num_samples):
        """Generate synthetic data."""
        return self.model.generate(count=num_samples).dataframe()

def get_generators():
    """
    Define synthetic data generators.
    Returns a dictionary mapping generator names to their functions.
    """
    return {
        'CTGAN': lambda: CTGAN(epochs=10),
        'TVAE': lambda: TVAESynthesizer(),
        'NFLOW': lambda: NFLOWSynthesizer()
    }
