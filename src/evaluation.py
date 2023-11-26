from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, \
    BinaryJaccardIndex, BinaryPrecision, BinaryRecall


class Evaluation():
    def __init__(self):
        # Instantiate all the metrics to be computed 
        self.accuracy = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.iou = BinaryJaccardIndex()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()


    def evaluate(self, gt_masks, pred_masks):
        # Update each metric result with the current batch
        self.accuracy.update(pred_masks, gt_masks)
        self.f1.update(pred_masks, gt_masks)
        self.iou.update(pred_masks, gt_masks)
        self.precision.update(pred_masks, gt_masks)
        self.recall.update(pred_masks, gt_masks)

    def evaluate_all(self, out_dir):
        # Compute from the metric object the global result
        with open(out_dir + 'evaluation.txt', 'a') as f:
            print(f'Accuracy: {self.accuracy.compute()}', file=f)
            print(f'F1: {self.f1.compute()}', file=f)
            print(f'IoU: {self.iou.compute()}', file=f)
            print(f'Precision: {self.precision.compute()}', file=f)
            print(f'Recall: {self.recall.compute()}', file=f)
