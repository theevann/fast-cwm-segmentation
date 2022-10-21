import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class TensorboardSummary(SummaryWriter):
    def __init__(self, log_dir, dataset):
        super().__init__(str(log_dir))
        self.dataset = dataset

    def write_metrics(self, metrics, epoch):
        for name, value in metrics.items():
            self.add_scalar('val/%s' % name, value, epoch)

    def visualize_image(self, image, target, output, global_step, n_im=3):
        with torch.no_grad():
            grid_image = make_grid(image[:n_im].cpu(), n_im, normalize=True)
            self.add_image('Image_train', grid_image, global_step)
            seg_map = torch.max(output[:n_im], 1)[1].cpu()
            grid_image = make_grid(self.decode_seg_map_sequence(seg_map), n_im, normalize=False, value_range=(0, 255))
            self.add_image('Predicted_label', grid_image, global_step)
            seg_map = target[:n_im].squeeze(1).cpu().long()
            grid_image = make_grid(self.decode_seg_map_sequence(seg_map), n_im, normalize=False, value_range=(0, 255))
            self.add_image('Groundtruth_label', grid_image, global_step)

    def decode_seg_map_sequence(self, label_masks):
        rgb_masks = [self.dataset.decode_segmap(label_mask) for label_mask in label_masks]
        return torch.stack(rgb_masks)
