import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.act(out)
        return out


class NNet(nn.Module):
    def __init__(self, channels, device):
        self.board_x = 6
        self.board_y = 7
        self.action_size = 7
        self.dropout = 0.3
        self.num_channels = channels

        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(2, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.bn4 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

        self.device = torch.device(device)
        self.to(device)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 2, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, board, player=None):
        encoded = self._encode(board, player)
        input = torch.from_numpy(encoded).reshape(1, 2, 6, 7).float().to(self.device)
        with torch.no_grad():
            return self.forward(input)

    def process_data(self, data, iters):
        self.train()
        board_states = np.empty((0, 2, 6, 7))
        policies = np.empty((0, 7))
        values = np.empty((0, 1))

        for experience in data:
            board = experience[0]
            player = experience[2] if len(experience) > 2 else 1
            board_channel1 = np.zeros((6, 7))   
            if player == 1:
                board_channel1[np.where(board == 1)] = 1
            else:
                board_channel1[np.where(board == 2)] = 1
            
            board_channel2 = np.zeros((6, 7))
            if player == 1:
                board_channel2[np.where(board == 2)] = 1
            else:
                board_channel2[np.where(board == 1)] = 1

            board_tensor = [np.reshape([board_channel1, board_channel2], [2, 6, 7])]
            board_states = np.append(board_states, board_tensor, axis=0)
            policies = np.append(policies, [experience[1]], axis=0)
            values = np.append(values, [[experience[3]]], axis=0)

        boards_tensor = torch.from_numpy(board_states).float().to(self.device)
        policies_tensor = torch.from_numpy(policies).float().to(self.device)
        values_tensor = torch.from_numpy(values).float().to(self.device)

        optimizer = optim.Adam(self.parameters())        
        value_criterion = nn.MSELoss()
        policy_criterion = nn.KLDivLoss()

        for i in range(iters):
            policies_predicted, values_predicted = self.forward(boards_tensor) 
            value_loss = value_criterion(values_predicted, values_tensor)
            policy_loss = policy_criterion(policies_predicted, policies_tensor)
            # policy_loss = -torch.sum(policies_tensor*policies_predicted)/policies_tensor.size()[0]
            total_loss = policy_loss + value_loss   
            print("v: %f p: %f" % (value_loss.item(), policy_loss.item()))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


class AlphaZeroNet(nn.Module):
    def __init__(self, out_channels, device="cuda"):
        super(AlphaZeroNet, self).__init__()
        # shared layers
        self.conv = nn.Conv2d(2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        # policy layers
        self.pconv1 = nn.Conv2d(out_channels, 2, 1, bias=False)
        self.pbn = nn.BatchNorm2d(2)
        self.pfc = nn.Linear(2 * 6 * 7, 7)
        #  # value layers
        self.vconv1 = nn.Conv2d(out_channels, 1, 1, bias=False)
        self.vbn = nn.BatchNorm2d(1)
        self.vfc1 = nn.Linear(6 * 7, out_channels, bias=False)
        self.vfc2 = nn.Linear(out_channels, 1)
        self.device = torch.device(device)
        self.to(self.device)

    def _encode(self, board, player=None):
        channel1 = np.zeros((6, 7), dtype=int)
        channel2 = np.zeros((6, 7), dtype=int)
        if player is None or player == 1:
            channel1[np.where(board == 1)] = 1
            channel2[np.where(board == 2)] = 1
        else:
            channel1[np.where(board == 2)] = 1
            channel2[np.where(board == 1)] = 1
        return np.append(channel1, channel2)

    # #2x1x1 conv -> bn -> relu -> fc
    def policy_head(self, x):
        out = self.pconv1(x) 
        out = self.pbn(out) 
        out = self.relu(out) 
        out = out.view(-1, 2 * 6 * 7)
        out = self.pfc(out) 
        return F.log_softmax(out, dim=1)
        
    # #1x1x1 conv -> bn -> relu -> fc -> relu -> fc -> tanh
    def value_head(self, x):
        out = self.vconv1(x) 
        out = self.vbn(out) 
        out = self.relu(out)
        out = out.view(-1, 1 * 6 * 7)
        out = self.vfc1(out) 
        out = self.relu(out)
        out = self.vfc2(out)
        return torch.tanh(out)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)
        return self.policy_head(out), self.value_head(out)

    def predict(self, board, player=None):
        channel1 = np.zeros((6, 7), dtype=int)
        channel2 = np.zeros((6, 7), dtype=int)
        if player is None or player == 1:
            channel1[np.where(board == 1)] = 1
            channel2[np.where(board == 2)] = 1
        else:
            channel1[np.where(board == 2)] = 1
            channel2[np.where(board == 1)] = 1
        input = torch.from_numpy(np.append(channel1, channel2)).reshape(1, 2, 6, 7).float().to(self.device)
        with torch.no_grad():
            return self.forward(input)

    def process_data(self, data, epochs=10, batch_size=64, lr=1e-3, weight_decay=0.0, minimal_logging=False):
        self.train()
        data = list(data)
        if not data:
            return None
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        value_criterion = nn.MSELoss()

        epoch_value_losses = []
        epoch_policy_losses = []
        batch_count = max(1, len(data) // batch_size)
        first_loss = None
        last_loss = None
        for epoch_idx in range(epochs):
            value_losses = []
            policy_losses = []
            for _ in range(batch_count):
                sample_ids = np.random.randint(len(data), size=batch_size)
                boards = np.empty((batch_size, 2, 6, 7), dtype=np.float32)
                policies = np.empty((batch_size, 7), dtype=np.float32)
                values = np.empty((batch_size, 1), dtype=np.float32)

                for i, idx in enumerate(sample_ids):
                    board, policy, player, value = data[idx]
                    encoded = self._encode(board, player)
                    boards[i] = np.reshape(encoded, [2, 6, 7])
                    policies[i] = policy
                    values[i] = value

                boards_tensor = torch.from_numpy(boards).float().to(self.device)
                policies_tensor = torch.from_numpy(policies).float().to(self.device)
                values_tensor = torch.from_numpy(values).float().to(self.device)

                policies_predicted, values_predicted = self.forward(boards_tensor)
                value_loss = value_criterion(values_predicted, values_tensor)
                policy_loss = -torch.sum(policies_tensor * policies_predicted) / policies_tensor.size(0)
                total_loss = policy_loss + value_loss
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                if first_loss is None:
                    first_loss = (policy_loss.item(), value_loss.item())
                last_loss = (policy_loss.item(), value_loss.item())
                optimizer.zero_grad()
                total_loss.backward()
                
                # Compute gradient norm
                grad_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                if 'grad_norms' not in dir():
                    grad_norms = []
                grad_norms.append(grad_norm)
                
                optimizer.step()
            epoch_value_losses.append(float(np.mean(value_losses)))
            epoch_policy_losses.append(float(np.mean(policy_losses)))
            if not minimal_logging:
                print(
                    "epoch %d/%d | batches=%d | Loss_pi=%.5f | Loss_v=%.5f"
                    % (epoch_idx + 1, epochs, batch_count, epoch_policy_losses[-1], epoch_value_losses[-1])
                )
        
        # Compute value prediction accuracy (how well values match targets)
        value_pred_error = float(np.mean(value_losses)) ** 0.5  # RMSE
        
        if epoch_value_losses and epoch_policy_losses:
            return {
                "value_loss": float(np.mean(epoch_value_losses)),
                "policy_loss": float(np.mean(epoch_policy_losses)),
                "start": first_loss,
                "end": last_loss,
                "grad_norm": float(np.mean(grad_norms)) if 'grad_norms' in dir() and grad_norms else 0.0,
                "value_pred_error": value_pred_error,
            }
        return None


class AlphaZeroResNet(nn.Module):
    def __init__(self, channels=128, num_blocks=6, device="cuda", dropout=0.3):
        super(AlphaZeroResNet, self).__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.device = torch.device(device)

        self.conv = nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        self.pconv = nn.Conv2d(channels, 2, 1, bias=False)
        self.pbn = nn.BatchNorm2d(2)
        self.pfc = nn.Linear(2 * 6 * 7, 7)

        self.vconv = nn.Conv2d(channels, 1, 1, bias=False)
        self.vbn = nn.BatchNorm2d(1)
        self.vfc1 = nn.Linear(6 * 7, channels)
        self.vfc2 = nn.Linear(channels, 1)

        self.to(self.device)

    def _encode(self, board, player=None):
        channel1 = np.zeros((6, 7), dtype=int)
        channel2 = np.zeros((6, 7), dtype=int)
        if player is None or player == 1:
            channel1[np.where(board == 1)] = 1
            channel2[np.where(board == 2)] = 1
        else:
            channel1[np.where(board == 2)] = 1
            channel2[np.where(board == 1)] = 1
        return np.append(channel1, channel2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        for block in self.res_blocks:
            out = block(out)

        p = self.pconv(out)
        p = self.pbn(p)
        p = self.act(p)
        p = p.view(-1, 2 * 6 * 7)
        p = F.dropout(p, p=self.dropout, training=self.training)
        p = self.pfc(p)

        v = self.vconv(out)
        v = self.vbn(v)
        v = self.act(v)
        v = v.view(-1, 6 * 7)
        v = self.vfc1(v)
        v = self.act(v)
        v = F.dropout(v, p=self.dropout, training=self.training)
        v = self.vfc2(v)

        return F.log_softmax(p, dim=1), torch.tanh(v)

    def predict(self, board, player=None):
        encoded = self._encode(board, player)
        input = torch.from_numpy(encoded).reshape(1, 2, 6, 7).float().to(self.device)
        with torch.no_grad():
            return self.forward(input)

    def process_data(self, data, epochs=10, batch_size=64, lr=1e-3, weight_decay=0.0, minimal_logging=False):
        self.train()
        data = list(data)
        if not data:
            return None
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        value_criterion = nn.MSELoss()

        epoch_value_losses = []
        epoch_policy_losses = []
        batch_count = max(1, len(data) // batch_size)
        first_loss = None
        last_loss = None
        for _ in range(epochs):
            value_losses = []
            policy_losses = []
            for _ in range(batch_count):
                sample_ids = np.random.randint(len(data), size=batch_size)
                boards = np.empty((batch_size, 2, 6, 7), dtype=np.float32)
                policies = np.empty((batch_size, 7), dtype=np.float32)
                values = np.empty((batch_size, 1), dtype=np.float32)

                for i, idx in enumerate(sample_ids):
                    board, policy, player, value = data[idx]
                    encoded = self._encode(board, player)
                    boards[i] = np.reshape(encoded, [2, 6, 7])
                    policies[i] = policy
                    values[i] = value

                boards_tensor = torch.from_numpy(boards).float().to(self.device)
                policies_tensor = torch.from_numpy(policies).float().to(self.device)
                values_tensor = torch.from_numpy(values).float().to(self.device)

                policies_predicted, values_predicted = self.forward(boards_tensor)
                value_loss = value_criterion(values_predicted, values_tensor)
                policy_loss = -torch.sum(policies_tensor * policies_predicted) / policies_tensor.size(0)
                total_loss = policy_loss + value_loss
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                if first_loss is None:
                    first_loss = (policy_loss.item(), value_loss.item())
                last_loss = (policy_loss.item(), value_loss.item())
                optimizer.zero_grad()
                total_loss.backward()
                
                # Compute gradient norm
                grad_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                if 'grad_norms' not in dir():
                    grad_norms = []
                grad_norms.append(grad_norm)
                
                optimizer.step()
            epoch_value_losses.append(float(np.mean(value_losses)))
            epoch_policy_losses.append(float(np.mean(policy_losses)))
            if not minimal_logging:
                print("v: %f p: %f" % (epoch_value_losses[-1], epoch_policy_losses[-1]))
        
        # Compute value prediction accuracy (how well values match targets)
        value_pred_error = float(np.mean(value_losses)) ** 0.5 if value_losses else 0.0  # RMSE
        
        if value_losses and policy_losses:
            return {
                "value_loss": float(np.mean(epoch_value_losses)),
                "policy_loss": float(np.mean(epoch_policy_losses)),
                "grad_norm": float(np.mean(grad_norms)) if 'grad_norms' in dir() and grad_norms else 0.0,
                "value_pred_error": value_pred_error,
                "start": first_loss,
                "end": last_loss,
            }
        return None
