import torch.nn as nn
import torch
from tqdm import tqdm


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    model_params_str, model_params = "", 0
    params_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_list.append(f"{name}/{list(param.size())}/{param.numel()}")

            model_params += param.numel()
    model_params_str = "\n ".join(params_list)
    return model_params_str, model_params


def custom_collate_fn_for_variable_seq_length(batch, word_to_idx, device):

    # Prepare the datapoints
    x, y = zip(*batch)

    # Pad x so that all the examples in the batch have the same size
    pad_token_ix = word_to_idx["<pad>"]
    x_tensor = [torch.LongTensor(x_i) for x_i in x]
    x_tensor = nn.utils.rnn.pad_sequence(
        x_tensor, batch_first=True, padding_value=pad_token_ix
    )

    y_tensor = torch.FloatTensor(y)
    y_tensor = y_tensor.unsqueeze(-1)

    return (
        x_tensor.to(device),
        y_tensor.to(device),
    )


class SentimentClassifierBaseline(nn.Module):

    def __init__(self, hyperparameters, vocab_size):
        super(SentimentClassifierBaseline, self).__init__()
        # Parameters
        self.embedding_dim = hyperparameters["embed_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.n_layers = hyperparameters["lstm_n_layers"]
        self.bidirectional = hyperparameters["lstm_bidirectional"]
        self.dropout_rate = hyperparameters["lstm_dropout_rate"]

        # Setting up the layers
        self.embeds = nn.Embedding(vocab_size, self.embedding_dim)
        self.hidden_layer = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
        )
        if 1 < self.n_layers:
            self.hidden_layer = nn.LSTM(
                self.embedding_dim,
                self.hidden_dim,
                num_layers=self.n_layers,
                bidirectional=self.bidirectional,
                dropout=self.dropout_rate,
            )
        self.output_layer = nn.Linear(
            2 * self.hidden_dim if self.bidirectional else self.hidden_dim, 1
        )
        self.probabilities = nn.Sigmoid()

    def forward(self, inputs_BL):
        """
        Notes for dimensions:
        - B: batch size
        - L: sequence length
        - E: embedding dimensions
        - H: LSTM hidden dimensions
        - S: score (dimension 1)
        """
        # print(["inputs_BL", inputs_BL.size()])
        embedded_sequences_BLE = self.embeds(inputs_BL)
        # print(["embedded_sequences_BLE", embedded_sequences_BLE.size()])
        output_rnn_BLH, (hidden_BLH, cell_BLH) = self.hidden_layer(
            embedded_sequences_BLE
        )
        # print(["output_rnn_BLH", output_rnn_BLH.size()])
        output_rnn_BS = output_rnn_BLH[:, -1, :]
        # print(["output_rnn_BLS", output_rnn_BS.size()])
        output_scores_BLS = self.output_layer(output_rnn_BS)
        # print(["output_scores_BLS", output_scores_BLS.size()])
        output_sigmoid_BLS = self.probabilities(output_scores_BLS)
        # print(["output_sigmoid_BLS", output_sigmoid_BLS.size()])
        return output_sigmoid_BLS


def bce_loss_function(batch_outputs, batch_labels):
    # print(["bce(). batch_outputs.size()", batch_outputs.size()])
    # print(["bce(). batch_labels.size()", batch_labels.size()])

    # Calculate the loss for the whole batch
    bceloss = nn.BCELoss()
    loss = bceloss(batch_outputs, batch_labels)
    # print(f"bce(). loss={loss}")
    return loss


def optimizer_obj(model, optimizer_name, learning_rate=0.001):
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    return torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_epoch_for_per_sequence(loss_function, optimizer, model, loader):

    # Clear the gradients
    optimizer.zero_grad()

    # Keep track of the total loss for the batch
    total_loss = 0
    for sequences_of_batch_BL, targets_of_batch_BS in tqdm(loader):
        # print(["sequences_of_batch_BL", sequences_of_batch_BL.size()])
        # print(["targets_of_batch_BS", targets_of_batch_BS.size()])
        # print(["lengths_of_batch", lengths_of_batch.size()])
        outputs_BC = model(sequences_of_batch_BL)
        # print(["outputs_BC", outputs_BC.size()])

        # Compute the batch loss
        loss = loss_function(
            outputs_BC,
            targets_of_batch_BS,
        )
        # Calculate the gradients
        loss.backward()
        # Update the parameters
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def train_sentiment_analysis(loss_function, optimizer, model, loader, num_epochs=10000):
    epoch_and_loss_list = [["epoch", "loss"]]
    print(f"num_epochs={num_epochs}")
    for epoch in range(num_epochs):
        epoch_loss = train_epoch_for_per_sequence(
            loss_function, optimizer, model, loader
        )
        if (epoch == 0) or ((epoch + 1) == num_epochs) or ((epoch + 1) % 100 == 0):
            print(f"epoch={(epoch+1)}, epoch_loss={epoch_loss}")
            epoch_and_loss_list.append([(epoch + 1), float(epoch_loss)])
    return epoch_and_loss_list


def custom_collate_fn(batch, device):
    """
    Dimension key:
     - B: Batch size
     - L: Length of sequence
     - C: Number of classes
    """
    # Prepare the datapoints
    x, y = zip(*batch)

    x_tensor = [torch.LongTensor(xi) for xi in x]
    x_tensor_BL = torch.stack(x_tensor)

    y_tensor = [torch.FloatTensor(yi) for yi in y]
    y_tensor_BC = torch.stack(y_tensor)

    lengths = [len(label) for label in y]
    lengths_tensor_B = torch.LongTensor(lengths)

    return x_tensor_BL.to(device), y_tensor_BC.to(device), lengths_tensor_B.to(device)


class WordWindowMulticlassClassifierBaseline(nn.Module):

    def __init__(self, hyperparameters, vocab_size, num_classes=2):
        super(WordWindowMulticlassClassifierBaseline, self).__init__()

        self.window_size = hyperparameters["window_size"]
        self.embed_dim = hyperparameters["embed_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.freeze_embeddings = hyperparameters["freeze_embeddings"]

        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        if self.freeze_embeddings:
            self.embedding.weight.requires_grad_ = False
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(self.embed_dim * (self.window_size * 2 + 1), self.hidden_dim),
            nn.ReLU(),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()
        )
        self.output_layer = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, inputs_BL):
        """
        Dimension key:
         - B: Batch size
         - E: Size of embedding
         - W: Size of word window embedding
         - H: Size of hidden layer embedding
         - L: Length of sequence
         - C: Number of classes
        """
        print(f"inputs_BL.size(): {inputs_BL.size()}")
        B, L = inputs_BL.size()

        embedded_windows_BLE = self.embedding(inputs_BL)
        # For debugging
        # print(f"embedded_windows_BLE.size(): {embedded_windows_BLE.size()}")

        extended_embedded_windows_BW = torch.reshape(
            embedded_windows_BLE, (B, (self.window_size * 2 + 1) * self.embed_dim)
        )
        # For debugging
        # print(
        # f"extended_embedded_windows_BW.size(): {extended_embedded_windows_BW.size()}"
        # )
        hidden1_BH = self.hidden_layer1(extended_embedded_windows_BW)
        hidden2_BH = self.hidden_layer2(hidden1_BH)
        output_BC = self.output_layer(hidden2_BH)
        softmax_BC = nn.functional.softmax(output_BC, dim=1)
        return softmax_BC


def ce_loss_function(batch_outputs, batch_labels):
    # :== Do not remove. Used for debugging
    # print(batch_outputs)
    # print(batch_outputs.shape)
    # print(batch_labels)
    # print(batch_labels.shape)
    # :== Do not remove. Used for debugging

    # Calculate the loss for the whole batch
    celoss = nn.CrossEntropyLoss()
    loss = celoss(batch_outputs, batch_labels)

    return loss


def train_epoch_for_per_token(loss_function, optimizer, model, loader):
    """
    Dimension key:
     - B: Batch size
     - L: Length of sequence
     - C: Number of classes
    """

    # Keep track of the total loss for the batch
    total_loss = 0
    for batch_inputs_BL, batch_labels_BC, batch_lengths_B in tqdm(loader):
        # print(batch_inputs) # for debugging
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        # print("batch_inputs_BL.size()")
        # print(batch_inputs_BL.size())
        outputs_BC = model.forward(batch_inputs_BL)
        # print("outputs_BC.size()")
        # print(outputs_BC.size())
        # Compute the batch loss
        # print("batch_labels_BC.size()")
        # print(batch_labels_BC.size())
        loss = loss_function(outputs_BC, batch_labels_BC)
        # Calculate the gradients
        loss.backward()
        # Update the parameters
        optimizer.step()
        # print("batch_lengths_B.size()")
        # print(batch_lengths_B.size())
        total_loss += loss.item() / batch_lengths_B[0]

    return total_loss


def train_named_entity_recognition(
    loss_function, optimizer, model, loader, num_epochs=10000
):
    epoch_and_loss_list = [["epoch", "loss"]]
    for epoch in range(num_epochs):
        epoch_loss = train_epoch_for_per_token(loss_function, optimizer, model, loader)
        if epoch % 10 == 0:
            print(f"epoch={epoch}, epoch_loss={epoch_loss}")
            epoch_and_loss_list.append([epoch, float(epoch_loss.numpy())])
    return epoch_and_loss_list
