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
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    t = list(set(text))
    t = sorted(t)
    indices = list(range(len(t)))
    char_to_idx = dict(zip(t, indices))
    idx_to_char = dict(zip(indices, t))
    # ========================
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
    # ====== YOUR CODE: ======
    n_removed = 0
    for char in chars_to_remove:
        n_removed += text.count(char)
        text = text.replace(char, '')
    text_clean = text
    # ========================
    return text_clean, n_removed


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
    # ====== YOUR CODE: ======
    zeros = torch.zeros((len(text), len(char_to_idx)), dtype=torch.int8)
    ones = torch.ones((len(text)), dtype=torch.int8)
    col_index = torch.tensor(list(map(lambda c: char_to_idx[c], text)))
    row_index = torch.arange(0, len(text)).view(len(text), 1)
    index_tensor = torch.cat((row_index, col_index.view(col_index.shape[0], 1)), 1)
    zeros.index_put_(tuple(index_tensor.t()), ones)
    result = zeros
    # ========================
    return result


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
    # ====== YOUR CODE: ======
    indices = torch.nonzero(embedded_text, as_tuple=False)
    result = ''.join(list(map(lambda i: idx_to_char[i[1].item()], indices)))
    # ========================
    return result


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
    # ====== YOUR CODE: ======
    n = len(text) // seq_len
    embedded = chars_to_onehot(text[:seq_len * n + 1], char_to_idx)
    samples = torch.stack(torch.split(embedded[:-1], seq_len))

    _, max_indices = torch.max(embedded[1:], dim=1)
    labels = torch.stack(torch.split(max_indices, seq_len))

    samples.to(device)
    labels.to(device)
    # ========================
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
    # ====== YOUR CODE: ======
    y_hat = y / temperature
    result = torch.softmax(y_hat, dim=dim)
    # ========================
    return result


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
    # ====== YOUR CODE: ======
    out_text = start_sequence
    with torch.no_grad():
        sample = chars_to_onehot(start_sequence, char_to_idx)
        sample = sample.unsqueeze(0)
        last_layer, hidden_state = model(sample.to(dtype=torch.float), hidden_state=None)

        for i in range(n_chars - len(start_sequence)):
            prob = hot_softmax(last_layer[0][-1], temperature=T)
            best_prob = torch.multinomial(prob, 1).item()
            next_char = idx_to_char[best_prob]
            out_text += next_char
            sample = chars_to_onehot(next_char, char_to_idx)
            sample = sample.unsqueeze(0)
            last_layer, hidden_state = model(sample.to(dtype=torch.float), hidden_state)
    # ========================

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
        idx = []  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        batches = len(self.dataset) // self.batch_size
        for i in range(batches):
            for j in range(self.batch_size):
                idx.append(i + j * batches)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


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
        # ====== YOUR CODE: ======
        for i in range(n_layers):
            if i == 0:
                layer_in_dim = in_dim
                layer_out_dim = h_dim
            elif i == n_layers:
                layer_in_dim = h_dim
                layer_out_dim = out_dim
            else:
                layer_in_dim = h_dim
                layer_out_dim = h_dim
            xz = nn.Linear(layer_in_dim, layer_out_dim)
            self.add_module(f'layer {i + 1} xz', xz)
            hz = nn.Linear(h_dim, layer_out_dim, bias=False)
            self.add_module(f'layer {i + 1} hz', hz)
            xr = nn.Linear(layer_in_dim, layer_out_dim)
            self.add_module(f'layer {i + 1} xr', xr)
            hr = nn.Linear(h_dim, layer_out_dim, bias=False)
            self.add_module(f'layer {i + 1} hr', hr)
            xg = nn.Linear(layer_in_dim, layer_out_dim)
            self.add_module(f'layer {i + 1} xg', xg)
            hg = nn.Linear(h_dim, layer_out_dim, bias=False)
            self.add_module(f'layer {i + 1} hg', hg)
            if dropout != 0:
                d = nn.Dropout2d(dropout)
                self.add_module(f'layer {i + 1} dropout', d)
            else:
                d = None
            layer_dict = {'xz': xz, 'hz': hz, 'xr': xr, 'hr': hr, 'xg': xg, 'hg': hg, 'dropout': d}
            self.layer_params.append(layer_dict)
        hy = nn.Linear(h_dim, out_dim)
        self.layer_params.append(hy)
        self.add_module('hy', hy)
        # ========================

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
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        self.to(input.device)
        T = input.shape[1]
        X = list(input.split(1, 1))

        for i in range(self.n_layers):

            for j in range(T):
                if i == 0:
                    X[j] = X[j].squeeze()
                z = torch.sigmoid(
                    self.layer_params[i]['xz'].forward(X[j]) +
                    self.layer_params[i]['hz'].forward(layer_states[i]))
                r = torch.sigmoid(
                    self.layer_params[i]['xr'].forward(X[j]) +
                    self.layer_params[i]['hr'].forward(layer_states[i]))
                g = torch.tanh(
                    self.layer_params[i]['xg'].forward(X[j]) +
                    self.layer_params[i]['hg'].forward(r * layer_states[i]))

                layer_states[i] = z * layer_states[i] + (1 - z) * g

                if self.layer_params[i]['dropout'] is not None:
                    X[j] = self.layer_params[i]['dropout'].forward(layer_states[i])
                else:
                    X[j] = layer_states[i]

        for i in range(T):
            X[i] = self.layer_params[-1].forward(X[i])
        layer_output = torch.stack(X, dim=1)
        hidden_state = torch.stack(layer_states, dim=1)
        # ========================
        return layer_output, hidden_state
