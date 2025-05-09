import torch

def _calculate_loss(model, distances, y_true, lambda_binary, lambda_L1):
    # label loss
    loss_cls = (distances*y_true).sum(axis=1)
    loss_cls = loss_cls.mean()

    # regularization loss
    loss_binary = model.binary_regularization()
    loss_sparsity = model.sparsity_regularization()

    # total loss
    return loss_cls + (lambda_binary * loss_binary) + (lambda_L1 * loss_sparsity)

def train_epoch(model, train_dataloader, optimizer, lambda_binary, lambda_L1, device='cpu'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in train_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        distances = model(x_batch)

        # total loss
        loss = _calculate_loss(model, distances, y_batch, lambda_binary, lambda_L1)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulative training loss
        total_loss += loss.item()

        # calculate the prediction accuracy
        predicted = distances.argmin(dim=1)
        real_labels = y_batch.argmax(dim=1)
        correct += (predicted == real_labels).sum().item()
        total += x_batch.size(0)

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

def val_epoch(model, val_dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in val_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            dist = model(x_batch)

            predicted = dist.argmin(dim=1)
            real_labels = y_batch.argmax(dim=1)
            correct += (predicted == real_labels).sum().item()
            total += x_batch.size(0)

    accuracy = correct / total * 100
    return accuracy