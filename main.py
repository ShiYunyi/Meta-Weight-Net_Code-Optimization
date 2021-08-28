import argparse
import torch.optim
# from torch.utils.tensorboard import SummaryWriter
from meta import *
from model import *
from noisy_long_tail_CIFAR import *
from utils import *


parser = argparse.ArgumentParser(description='Meta_Weight_Net')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=1e-5)
parser.add_argument('--meta_weight_decay', type=float, default=0.)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--imbalanced_factor', type=int, default=None)
parser.add_argument('--corruption_type', type=str, default=None)
parser.add_argument('--corruption_ratio', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--max_epoch', type=int, default=120)

parser.add_argument('--meta_interval', type=int, default=1)
parser.add_argument('--paint_interval', type=int, default=20)

args = parser.parse_args()
print(args)


def meta_weight_net():
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
#     writer = SummaryWriter(log_dir='.\\mwn')

    meta_net = MLP(hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers).to(device=args.device)
    net = ResNet32(args.dataset == 'cifar10' and 10 or 100).to(device=args.device)

    criterion = nn.CrossEntropyLoss().to(device=args.device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    lr = args.lr

    train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list = build_dataloader(
        seed=args.seed,
        dataset=args.dataset,
        num_meta_total=args.num_meta,
        imbalanced_factor=args.imbalanced_factor,
        corruption_type=args.corruption_type,
        corruption_ratio=args.corruption_ratio,
        batch_size=args.batch_size,
    )

    meta_dataloader_iter = iter(meta_dataloader)
#     with torch.no_grad():
#         for point in range(500):
#             x = torch.tensor(point / 10).unsqueeze(0).to(args.device)
#             fx = meta_net(x)
#             writer.add_scalar('Initial Meta Net', fx, point)

    for epoch in range(args.max_epoch):

        if epoch >= 80 and epoch % 20 == 0:
            lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        print('Training...')
        for iteration, (inputs, labels) in enumerate(train_dataloader):
            net.train()
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            if (iteration + 1) % args.meta_interval == 0:
                pseudo_net = ResNet32(args.dataset == 'cifar10' and 10 or 100).to(args.device)
                pseudo_net.load_state_dict(net.state_dict())
                pseudo_net.train()

                pseudo_outputs = pseudo_net(inputs)
                pseudo_loss_vector = functional.cross_entropy(pseudo_outputs, labels.long(), reduction='none')
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
                pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
                pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)

                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                del pseudo_grads

                try:
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    meta_inputs, meta_labels = next(meta_dataloader_iter)

                meta_inputs, meta_labels = meta_inputs.to(args.device), meta_labels.to(args.device)
                meta_outputs = pseudo_net(meta_inputs)
                meta_loss = criterion(meta_outputs, meta_labels.long())

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            outputs = net(inputs)
            loss_vector = functional.cross_entropy(outputs, labels.long(), reduction='none')
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

            with torch.no_grad():
                weight = meta_net(loss_vector_reshape)

            loss = torch.mean(weight * loss_vector_reshape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Computing Test Result...')
        test_loss, test_accuracy = compute_loss_accuracy(
            net=net,
            data_loader=test_dataloader,
            criterion=criterion,
            device=args.device,
        )
#         writer.add_scalar('Loss', test_loss, epoch)
#         writer.add_scalar('Accuracy', test_accuracy, epoch)

        print('Epoch: {}, (Loss, Accuracy) Test: ({:.4f}, {:.2%}) LR: {}'.format(
            epoch,
            test_loss,
            test_accuracy,
            lr,
        ))

#         if (epoch + 1) % args.paint_interval == 0:
#             with torch.no_grad():
#                 for point in range(500):
#                     x = torch.tensor(point / 10).unsqueeze(0).to(args.device)
#                     fx = meta_net(x)
#                     writer.add_scalar('Meta Net of Epoch {}'.format(epoch), fx, point)

#     writer.close()


if __name__ == '__main__':
    meta_weight_net()
