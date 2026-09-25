from pydantic import BaseModel
from data_utils.constants import NOF_CONTEXTS, ContinualLearningMethodClass, DataSetNameClass, DATA_SET_NAME_LIST

class DotProductDTO(BaseModel):
    cos_sim: float
    magnitude_ratio: float
    projection_loss: float

class TrainingResultDTO(BaseModel):
    method_used: ContinualLearningMethodClass

    nof_context: int
    datasets: list[DataSetNameClass]

    number_of_epochs: int
    training_time: float

    train_losses: list[float] = []
    train_accuracies: list[float] = []

    context_f1s: list[float]
    so_far_f1s: list[float]

    context_accuracies: list[float]
    so_far_accuracies: list[float]

    final_test_f1: float
    final_test_accuracy: float

    dot_products: dict[int, list[DotProductDTO]] = {}  # Mapping from context index to dot product value

    # confusion_matrix: list[list[int]]