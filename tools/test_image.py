import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datatools.test_dataset import TestDataset
from data.datatools.transforms import Denormalize
from utilities import get_preds

orientation_labels = [
    'front', 'rear', 'left', 'left front', 'left rear', 'right', 'right front',
    'right rear'
]

orientation_to_keypoints = {
    0: [11, 12, 7, 8, 9, 13, 14],
    1: [18, 16, 15, 19, 17, 11, 12],
    2: [8, 1, 11, 14, 15, 2, 17],
    3: [9, 14, 6, 8, 11, 1, 5],
    4: [2, 17, 15, 11, 14, 19, 1],
    5: [7, 3, 12, 13, 16, 4, 18],
    6: [9, 13, 5, 7, 12, 3, 16],
    7: [3, 4, 12, 16, 18, 19, 13]
}
orientation_to_keypoints = {
    k: [vi - 1 for vi in v]
    for k, v in orientation_to_keypoints.items()
}

denormalize = Denormalize()


def visualize_results(outputs, orientations, inputs, global_index):
    for index in range(outputs.shape[0]):
        output = outputs[index].cpu().numpy()
        image = inputs[index].cpu().numpy().transpose(1, 2, 0)
        image = denormalize(image)
        image = image.astype(np.uint8).copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for i in range(outputs.shape[1]):
            if i not in orientation_to_keypoints[int(orientations[index])]:
                continue
            coord = (int(output[i][0] / 56.0 * 224),
                     int(output[i][1] / 56.0 * 224))
            image = cv2.circle(image, coord, 5, (255, 0, 0), 2)
            image = cv2.putText(image, str(i + 1), coord,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                                cv2.LINE_AA)
            image = cv2.putText(image, orientation_labels[orientations[index]],
                                (0, image.shape[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite('/tmp/{:03d}_{:03d}.jpg'.format(global_index, index),
                    image)


def test_image(args, net):
    test_set = TestDataset(root=args.test_image_dir)
    test_loader = DataLoader(test_set,
                             shuffle=False,
                             batch_size=args.test_batch_size,
                             num_workers=args.num_workers)

    if args.use_case == 'stage1':
        with torch.no_grad():
            with tqdm(total=len(test_loader),
                      ncols=0,
                      file=sys.stdout,
                      desc='Stage 1 Evaluation...') as pbar:
                for i, in_batch in enumerate(test_loader):
                    image_in1, _ = in_batch
                    if torch.cuda.is_available():
                        image_in1 = image_in1.cuda()
                    coarse_kp = net(image_in1)
                    print('coarse_kp: {}'.format(get_preds(coarse_kp).shape))
                    pbar.update()

    elif args.use_case == 'stage2':
        with torch.no_grad():
            with tqdm(total=len(test_loader),
                      ncols=0,
                      file=sys.stdout,
                      desc='Stage 2 Evaluation...') as pbar:
                for i, in_batch in enumerate(test_loader):
                    image_in1, image_in2 = in_batch
                    if torch.cuda.is_available():
                        image_in1, image_in2 = \
                            image_in1.cuda(), image_in2.cuda()

                    coarse_kp, fine_kp, orientation = net(image_in1, image_in2)

                    predicted_keypoints = get_preds(fine_kp)
                    print('fine_kp: {}'.format(fine_kp.shape))
                    print('predicted_keypoints: {}'.format(
                        predicted_keypoints.shape))

                    _, predicted_orientation = torch.max(orientation.data, 1)
                    print('predicted_orientation: {}'.format(
                        predicted_orientation.shape))

                    if args.visualize:
                        visualize_results(predicted_keypoints,
                                          predicted_orientation, image_in1, i)

                    pbar.update()
