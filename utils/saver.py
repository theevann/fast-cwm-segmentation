import shutil
from pathlib import Path

import torch


class Saver(object):

    def __init__(self, args):
        self.args = args
        folder_name = args.archi
        self.directory = Path(args.save_dir) / folder_name / args.checkname
        self.runs = sorted(self.directory.glob("experiment_*"),
                           key=lambda x: int(str(x).split('_')[-1]))

        run_id = int(str(self.runs[-1]).split('_')[-1]) + 1 if self.runs else 0
        self.experiment_dir = Path(args.xp_folder or (self.directory / 'experiment_{}'.format(run_id)))
        self.experiment_dir.mkdir(exist_ok=True, parents=True)

        if self.runs:
            previous_mious = [0.0]
            for run in self.runs:
                run_id = str(run).split('_')[-1]
                path = self.directory / 'experiment_{}'.format(run_id) / 'best_pred.txt'
                if path.exists():
                    miou = float(path.read_text())
                    previous_mious.append(miou)
            self.max_miou = max(previous_mious)

    def save_experiment_config(self):
        with (self.experiment_dir / 'parameters.txt').open('w') as log_file:
            for key, val in vars(self.args).items():
                log_file.write(key + ':' + str(val) + '\n')

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        filename = str(self.experiment_dir / filename)
        self.args.logger.info('Saving model to %s' % filename)
        torch.save(state, filename)

        if is_best:
            best_pred = state['best_pred']
            (self.experiment_dir / 'best_pred.txt').write_text(str(best_pred))
            if (not self.runs) or (best_pred > self.max_miou):
                shutil.copyfile(filename, str(self.experiment_dir / "best.pth" ))
                shutil.copyfile(filename, str(self.directory / 'model_best.pth'))

