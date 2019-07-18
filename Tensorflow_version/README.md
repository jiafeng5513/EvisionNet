### Notes
After adding data augmentation and removing batch normalization (along with some other minor tweaks), we have been able to train depth models better than what was originally reported in the paper even without using additional Cityscapes data or the explainability regularization. The provided pre-trained model was trained on KITTI only with smooth weight set to 0.5, and achieved the following performance on the Eigen test split (Table 1 of the paper):

| Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|---------|--------|-------|-----------|-------|-------|-------|
| 0.183   | 1.595  | 6.709 | 0.270     | 0.734 | 0.902 | 0.959 | 

When trained on 5-frame snippets, the pose model obtains the following performanace on the KITTI odometry split (Table 3 of the paper):

| Seq. 09            | Seq. 10            |
|--------------------|--------------------|
| 0.016 (std. 0.009) | 0.013 (std. 0.009) |
