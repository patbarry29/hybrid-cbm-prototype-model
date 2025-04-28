import time

import torch
from src.utils import *

def _get_outputs(model, inputs):
    outputs = model(inputs)
    if len(outputs) == 2:
        return outputs[0], outputs[1]
    return outputs, None

def _calculate_concept_loss(concept_idx, main_output, aux_output, target, criterion, is_training, use_aux):
    # Process main output
    main_output_squeezed = main_output.squeeze()  # Shape [N]
    loss = criterion(main_output_squeezed, target)

    # Add auxiliary loss if needed
    if is_training and use_aux and aux_output is not None:
        aux_output_squeezed = aux_output[concept_idx].squeeze()  # Shape [N]
        aux_loss = criterion(aux_output_squeezed, target)
        loss += 0.4 * aux_loss  # Add weighted auxiliary loss

    return loss, main_output

def _log_progress(start_time, is_training, batch_idx, log_interval, loader, loss_meter, acc_meter):
    if is_training and (batch_idx + 1) % log_interval == 0:
        elapsed_time = time.time() - start_time
        print(f' Batch: {batch_idx+1:3d}/{len(loader)} | Avg. Loss: {loss_meter.avg:.4f} |'
            f' Avg. Acc.: {acc_meter.avg:.3f} | Time: {elapsed_time:.2f}s')
        return time.time() # Reset timer
    return start_time

def run_epoch_x_to_c(model, loader, criterion_list,  optimizer, n_concepts, is_training=False, use_aux=False, device='cpu', log_interval=50):
    """
    Modified run_epoch focused on X -> C training.
    criterion_list: List of loss functions for each concept.
    """
    if is_training:
        model.train() # use dropout layers and calculates gradients
    else:
        model.eval() # sets layers like BatchNorm to use running statistics

    # track average loss and accuracy throughout the epoch
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    start_time = time.time()

    for batch_idx, data in enumerate(loader):
        inputs = data[0].to(device)
        concept_labels = data[1].to(device)

        if is_training:
            optimizer.zero_grad()

        # Forward pass
        main_outputs, aux_outputs = _get_outputs(model, inputs)

        # Loss calculation
        total_loss = 0
        all_concept_outputs = [] # To store tensors for accuracy calculation

        for i in range(n_concepts):
            target_concepts = concept_labels[:, i].float()

            loss_i, _ = _calculate_concept_loss(
                i, main_outputs[i], aux_outputs, target_concepts,
                criterion_list[i], is_training, use_aux
            )

            total_loss += loss_i
            all_concept_outputs.append(main_outputs[i])

        # Average loss over attributes in the batch
        avg_batch_loss = total_loss / n_concepts

        # Backward pass and optimization
        if is_training:
            avg_batch_loss.backward()
            optimizer.step()

        # Accuracy Calculation (using the main outputs collected)
        sigmoid_outputs = torch.sigmoid(torch.cat(all_concept_outputs, dim=1))
        acc = binary_accuracy(sigmoid_outputs, concept_labels.int())

        # Update meters
        loss_meter.update(avg_batch_loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))

        # Logging
        start_time = _log_progress(start_time, is_training, batch_idx,
                                log_interval, loader, loss_meter, acc_meter)

    return loss_meter.avg, acc_meter.avg

# TO DO - look if model and optmiiser should be passed as param or by reference