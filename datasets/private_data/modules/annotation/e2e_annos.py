from .base import AnnotationBase
from datasets.private_data.utils.functional import initialize_object

class E2EAnnotations(AnnotationBase):
    def __init__(
        self,
        loader_output,
        mode,
        annotation_cfgs,
        sub_level_annos=False,
    ) -> None:
        super(E2EAnnotations, self).__init__(loader_output, mode)
        self.tasks = dict()
        self.sub_level_annos = sub_level_annos
        assert all(
            issubclass(ann_cfg["type"], AnnotationBase) for ann_cfg in annotation_cfgs.values()
        ), "Annotation type should be derived class of AnnotationBase"

        for task, config in annotation_cfgs.items():
            config.update({"loader_output": loader_output, "mode": mode})
            self.tasks[task] = initialize_object(config)

    def __getitem__(self, index):

        data_dict = dict()
        for task, ann_reader in self.tasks.items():
            result = ann_reader[index]
            if result is None:
                return None
            if self.sub_level_annos:
                data_dict[task] = result
            else:
                data_dict.update(result)

        return data_dict
