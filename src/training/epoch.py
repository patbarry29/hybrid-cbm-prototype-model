import time

import torch
from tqdm import tqdm
from src.utils import *

def _get_outputs(model, inputs):
    outputs = model(inputs)
    if len(outputs) == 2:
        return outputs[0], outputs[1]
    return outputs, None

def _calculate_concept_loss(concept_idx, main_output, aux_output, target, criterion,
                            is_training, use_aux):
    # Process main output
    main_output_squeezed = main_output.squeeze()  # Shape [N]
    loss = criterion(main_output_squeezed, target)

    # Add auxiliary loss if needed
    if is_training and use_aux and aux_output is not None:
        aux_output_squeezed = aux_output[concept_idx].squeeze()  # Shape [N]
        aux_loss = criterion(aux_output_squeezed, target)
        loss += 0.4 * aux_loss  # Add weighted auxiliary loss

    return loss, main_output

def run_epoch_x_to_c(model, loader, criterion_list,  optimizer, n_concepts,
                    is_training=False, use_aux=False, device='cpu', verbose=False,
                    return_outputs=None):
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

    outputs = []

    desc = "Training" if is_training else "Validation"
    tqdm_loader = tqdm(loader, desc=desc, leave=False, disable=not verbose) # Wrap the loader with tqdm

    for batch_idx, data in enumerate(tqdm_loader):
        inputs = data[0].to(device)
        concept_labels = data[1].to(device)

        if is_training:
            optimizer.zero_grad()

        # Forward pass
        main_outputs, aux_outputs = _get_outputs(model, inputs)
        if return_outputs=='main':
            outputs.append(main_outputs)

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
        if return_outputs=='sigmoid':
            outputs.append(sigmoid_outputs)
        acc = binary_accuracy(sigmoid_outputs, concept_labels.int())

        # Update meters
        loss_meter.update(avg_batch_loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))

        # Update tqdm progress bar description with current loss and accuracy
        tqdm_loader.set_postfix(loss=f'{loss_meter.avg:.4f}', acc=f'{acc_meter.avg:.4f}')

    if return_outputs is not None:
        return loss_meter.avg, acc_meter.avg, outputs
    return loss_meter.avg, acc_meter.avg

