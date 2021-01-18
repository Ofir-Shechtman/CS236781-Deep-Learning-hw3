import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    chars = list(set(text))
    chars.sort()
    char_to_idx, idx_to_char = dict(), dict()
    for i, char in enumerate(chars):
        char_to_idx[char] = i
        idx_to_char[i] = char
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    if chars_to_remove:
        text_clean = text.replace(chars_to_remove[0], ' ')
    else:
        return text, 0
    for char in chars_to_remove[1:]:
        text_clean = text_clean.replace(char, ' ')
    return text_clean, len(chars_to_remove)


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    idx_tensor = torch.tensor(list(map(lambda x: char_to_idx.get(x), list(text))))
    result = nn.functional.one_hot(idx_tensor, num_classes=len(char_to_idx))
    return result.type(torch.int8)


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    one_hot_encode = torch.argmax(embedded_text, dim=1, keepdim=False)
    char_list = list(map(lambda x: idx_to_char.get(x), one_hot_encode.tolist()))
    return ''.join(char_list)


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    samples_chars = [text[i * seq_len:(i + 1) * seq_len] for i in range(len(text) // seq_len)]
    labels_chars = [text[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(len(text) // seq_len)]
    samples = torch.stack(list(map(lambda seq: chars_to_onehot(seq, char_to_idx).to(device), samples_chars)))
    def seq_to_labels(seq):
        return torch.tensor(list(map(lambda x: char_to_idx.get(x), list(seq))), device=device)

    labels = torch.stack(list(map(lambda seq: seq_to_labels(seq), labels_chars)))
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    yt = y / temperature
    softmax = nn.Softmax(dim=dim)
    return softmax(yt)


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    with torch.no_grad():
        y, h_s = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(0), None
    for i in range(n_chars - len(start_sequence)):
        y_out, h_s = model(y.to(dtype=torch.float, device=device), h_s)
        char = idx_to_char.get(torch.multinomial(hot_softmax(y_out[0][-1], temperature=T), num_samples=1).item())
        sample = chars_to_onehot(char, char_to_idx)
        y = sample.unsqueeze(0)
        out_text += char

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        number_of_batches = len(self.dataset) // self.batch_size
        batch0 = [i * number_of_batches for i in range(self.batch_size)]

        get_j_batch = lambda batch0, j: [i + j for i in batch0]
        batches = [get_j_batch(batch0, j) for j in range(number_of_batches)]
        idx = [item for batch in batches for item in batch]  # idx should be a 1-d list of indices.
        return iter(idx)

    def __len__(self):
        return len(self.dataset) - len(self.dataset) % self.batch_size


class GRULayer(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, out_dim):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and out_dim > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_params = []

        self.add_module('Linear_Wxz', nn.Linear(in_features=in_dim, out_features=out_dim, bias=True))
        self.add_module('Linear_Wxr', nn.Linear(in_features=in_dim, out_features=out_dim, bias=False))
        self.add_module('Linear_Wxg', nn.Linear(in_features=in_dim, out_features=out_dim, bias=True))
        self.add_module('Linear_Whz', nn.Linear(in_features=out_dim, out_features=out_dim, bias=False))
        self.add_module('Linear_Whr', nn.Linear(in_features=out_dim, out_features=out_dim, bias=True))
        self.add_module('Linear_Whg', nn.Linear(in_features=out_dim, out_features=out_dim, bias=False))

    def forward(self, input: Tensor, hidden_state: Tensor):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state (for the first
        char). Shape should be (B, H) where B is the batch size
        and H is the number of hidden dimensions.
        :return: The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, H)  as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_input = input
        layer_states = []

        h = hidden_state
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()

        for char in range(seq_len):
            z = sigmoid(self.Linear_Wxz(layer_input[:, char, :]) + self.Linear_Whz(h))
            r = sigmoid(self.Linear_Wxr(layer_input[:, char, :]) + self.Linear_Whr(h))
            g = tanh(self.Linear_Wxg(layer_input[:, char, :]) + self.Linear_Whg(r * h))
            h = z * h + (1 - z) * g
            layer_states.append(h)
        return torch.stack(layer_states, dim=1)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        self.layers = []
        for i in range(n_layers):
            layer_in_dim = in_dim if i == 0 else h_dim
            layer = GRULayer(in_dim=layer_in_dim, out_dim=h_dim)
            self.add_module('Layer{}'.format(i), layer)
            dropout_layer = None
            if dropout:
                dropout_layer = nn.Dropout2d(p=dropout)
            self.layers.append((layer, dropout_layer))
        self.add_module('Linear_Why', nn.Linear(in_features=h_dim, out_features=out_dim, bias=True))

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape
        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.

        final_h = []
        for i, (layer, dropout) in enumerate(self.layers):
            layer_input = layer(layer_input, layer_states[i])
            if dropout is not None:
                layer_input = dropout(layer_input)
            final_h.append(layer_input[:, -1, :])
        layer_output = layer_input
        l=[]
        for t in torch.unbind(layer_output, dim=0):
            l.append(self.Linear_Why(t)
                     
        #self.Linear_Why(layer_output) 

        return torch.stack(l, dim=1), torch.stack(final_h, dim=1)
