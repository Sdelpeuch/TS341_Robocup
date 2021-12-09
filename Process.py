class Process:
    def __init__(self, dataProcessing):
        self.dataProcessing = dataProcessing

    def process(self):
        if self.dataProcessing.image.base_goal_1 == (-1, -1) and self.dataProcessing.image.base_goal_2 == (-1, -1):
            return
        elif (self.dataProcessing.image.base_goal_1 != (-1, -1) and self.dataProcessing.image.base_goal_2 == (-1, -1)) or (
                self.dataProcessing.image.base_goal_1 == (-1, -1) and self.dataProcessing.image.base_goal_2 != (-1, -1)):
            self.dataProcessing.segmentation_post()
        elif self.dataProcessing.image.base_goal_1 != (-1, -1) and self.dataProcessing.image.base_goal_2 != (-1, -1):
            self.dataProcessing.segmentation_goal()
