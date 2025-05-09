import torch
import torch.nn as nn

class PrototypeClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.protoypes = nn.Parameter(torch.rand(num_classes, num_features))  # initialize the prototype matrix P

    def forward(self, x):
        # x: (batch_size, num_features)
        # L1distance：|x_i - M_m|_1
        # (batch_size, num_classes, num_features)
        dist = torch.abs(x.unsqueeze(1) - torch.sigmoid(self.protoypes))
        dist = dist.sum(dim=2)
        return dist  # (batch_size, num_classes)

    def binary_regularization(self):
        sigmoid_protos = torch.sigmoid(self.protoypes)
        return (sigmoid_protos * (1 - sigmoid_protos)).mean()

    def sparsity_regularization(self):
        return torch.sum(torch.sigmoid(self.protoypes))

    def predict(self, x):
        with torch.no_grad():
            Prototypes = torch.sigmoid(self.protoypes)
            Prototypes[Prototypes>=0.5] = 1
            Prototypes[Prototypes<0.5]= 0
            dists = torch.abs(x.unsqueeze(1) - Prototypes)
            dists = dists.sum(dim=2)
            predictions = dists.argmin(dim=1)
        return predictions

    def get_sigmoid_prototypes(self):
        return torch.sigmoid(self.protoypes)

    def concept_wise_dist(self, x):
        with torch.no_grad():
            Prototypes = torch.sigmoid(self.protoypes)
            Prototypes[Prototypes>=0.5] = 1
            Prototypes[Prototypes<0.5]= 0
            dists = x.unsqueeze(1) - Prototypes
            # predictions = self.predict(x)
            # dists = dists[torch.arange(x.shape[0]), predictions,:]
        return dists

    def threshold(self, val_x, val_y):
        pass

    def conformal_predict(self, x):
        pass

    def explanation(self, x):
        pass





# class PrototypeLearner(nn.Module):
#     def forward(self, C_hat, Y_true=None, lambda_bin=0.1, lambda_spars=0.01):
#         # Get continuous prototypes
#         prototypes = self.prototypes.weight
#         prototypes_sigmoid = torch.sigmoid(prototypes)  # Shape: [num_classes, num_concepts]

#         # Calculate absolute difference between concepts and prototypes
#         concept_distances = torch.abs(C_hat.unsqueeze(1) - prototypes_sigmoid)  # Shape: [batch_size, num_classes, num_concepts]
#         # Sum distances across concept dimension
#         label_distances = concept_distances.sum(dim=2)  # Shape: [batch_size, num_classes]
#         pred_label = label_distances.argmin(dim=1)  # Shape: [batch_size]

#         # Classification loss - using the distances for labeled classes
#         loss_class = torch.mean(torch.sum(label_distances * Y_true, dim=1))

#         # Binarization loss - encourages prototypes to be binary (0 or 1)
#         loss_bin = torch.mean(prototypes_sigmoid * (1 - prototypes_sigmoid))

#         # Sparsity loss - encourages fewer active concepts
#         loss_spars = torch.mean(torch.abs(prototypes_sigmoid))

#         # Combine losses
#         total_loss = loss_class + (lambda_bin * loss_bin) + (lambda_spars * loss_spars)

#         return pred_label, total_loss

#     def get_binary_prototypes(self):
#         with torch.no_grad():
#             binary_prototypes = (torch.sigmoid(self.prototypes.weight) > 0.5).float()
#         return binary_prototypes

#     def get_sigmoid_prototypes(self):
#         return torch.sigmoid(self.prototypes.weight)