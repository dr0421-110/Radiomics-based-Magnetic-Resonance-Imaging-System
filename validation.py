import torch
import time
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              distributed=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    cls0_acc=AverageMeter()
    cls1_acc = AverageMeter()
    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets,clinic,name) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            inputs=inputs.to(device, non_blocking=True)
            clinic = clinic.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs,clinic)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            acc0 = calculate_accuracy(outputs[targets==0], targets[targets==0])
            acc1 = calculate_accuracy(outputs[targets == 1], targets[targets == 1])
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            cls0_acc.update(acc0, targets[targets == 0].size(0))
            cls1_acc.update(acc1, targets[targets == 1].size(0))


            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.avg:.3f} ({acc.avg:.3f})\t'
                  'Acc0 {acc0.avg:.3f} ({acc0.avg:.3f})\t'
                  'Acc1 {acc1.avg:.3f} ({acc1.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies,
                      acc0=cls0_acc,
                      acc1=cls1_acc))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)

    return losses.avg
