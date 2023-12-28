"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import random

import minitorch


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        # TODO: Implement for Task 1.5.
        self.hidden_layers = hidden_layers
        self.layer_1 = Linear(2, 2)
        for idx in range(2, self.hidden_layers + 1):
            layer_name = f"layer_{idx}"
            setattr(self, layer_name, Linear(2, 2))
        self.final = Linear(2, 1)

    def forward(self, x):
        middle = [h.relu() for h in self.layer_1.forward(x)]
        intermediate = middle
        if self.hidden_layers >= 2:
            for idx in range(2, self.hidden_layers):
                layer_name = f"layer_{idx}"
                layer = getattr(self, layer_name)
                intermediate = [h.relu() for h in layer.forward(intermediate)]
        return self.final.forward(intermediate)[0].sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        output = []
        for i_idx in range(len(self.weights[0])):
            node_output = self.bias[i_idx].value
            for j_idx in range(len(self.weights)):
                node_output += inputs[i_idx] * self.weights[j_idx][i_idx].value
            output.append(node_output)

        return output


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        self.model.train()
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data

            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    data = minitorch.datasets["Xor"](PTS)
    HIDDEN = 10
    RATE = 0.5
    ScalarTrain(HIDDEN).train(data, RATE)
