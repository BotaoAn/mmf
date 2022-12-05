import torch 

# We will inherit the FastText Processor already present in MMF
from mmf.datasets.processors import FastTextProcessor
# registry is needed to register processor and model to be MMF discoverable
from mmf.common.registry import registry


# Register the processor so that MMF can discover it
@registry.register_processor("fasttext_sentence_vector")
class FastTextSentenceVectorProcessor(FastTextProcessor):
    # Override the call method
    def __call__(self, item):
        # This function is present in FastTextProcessor class and loads
        # fasttext bin
        self._load_fasttext_model(self.model_file)
        if "text" in item:
            text = item["text"]
        elif "tokens" in item:
            text = " ".join(item["tokens"])

        # Get a sentence vector for sentence and convert it to torch tensor
        sentence_vector = torch.tensor(
            self.model.get_sentence_vector(text),
            dtype=torch.float
        )

        # Return back a dict
        return {
            "text": sentence_vector
        }
    
    # Make dataset builder happy, return a random number
    def get_vocab_size(self):
        return None