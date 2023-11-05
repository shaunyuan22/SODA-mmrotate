# The Evaluation about SODA-A Dataset

With regard to the evaluation, we'd like to bring two important points to your attention:
 - The evaluation is performed on the original images (**NOT ON** the splitted images).
 - The `ignore` regions will not be used in the evaluation phase.

Hence you need to filter `ignore` annotations of the original json files in the rawData directory (i.e., `AnnsWithIgnore`) to get available json files stored in `Annotations` for final performance evaluation. Finally, you may have the following folder sturcture:

```none
SODA-A
├── rawData
│   ├── train
│   │   ├── Images
│   │   ├── Annotations
│   │   ├── AnnsWithIgnore
│   ├── val
│   │   ├── Images
│   │   ├── Annotations
│   │   ├── AnnsWithIgnore
│   ├── test
│   │   ├── Images
│   │   ├── Annotations
│   │   ├── AnnsWithIgnore
├── divData
│   ├── train
│   │   ├── Images
│   │   ├── Annotations
│   ├── val
│   │   ├── Images
│   │   ├── Annotations
│   ├── test
│   │   ├── Images
│   │   ├── Annotations
```
