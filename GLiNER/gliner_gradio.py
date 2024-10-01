from typing import Dict, Union
from gliner import GLiNER
import gradio as gr

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

examples = [
    
    [
        """
* Data Scientist, Data Analyst, or Data Engineer with 1+ years of experience.
* Experience with technologies such as Docker, Kubernetes, or Kubeflow
* Machine Learning experience preferred
* Experience with programming languages such as Python, C++, or SQL preferred
* Experience with technologies such as Databricks, Qlik, TensorFlow, PyTorch, Python, Dash, Pandas, or NumPy preferred
* BA or BS degree
* Active Secret OR Active Top Secret or Active TS/SCI clearance
""",
        "software package, programing language, software tool, degree, job title",
        0.3,
        False,
    ],
    
]


def ner(
    text, labels: str, threshold: float, nested_ner: bool
) -> Dict[str, Union[str, int, float]]:
    labels = labels.split(",")
    return {
        "text": text,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": 0,
            }
            for entity in model.predict_entities(
                text, labels, flat_ner=not nested_ner, threshold=threshold
            )
        ],
    }


with gr.Blocks(title="GLiNER-M-v2.1") as demo:
    gr.Markdown(
        """
        # GLiNER-base
        GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.
        """
    )
    
    input_text = gr.Textbox(
        value=examples[0][0], label="Text input", placeholder="Enter your text here"
    )
    with gr.Row() as row:
        labels = gr.Textbox(
            value=examples[0][1],
            label="Labels",
            placeholder="Enter your labels here (comma separated)",
            scale=2,
        )
        threshold = gr.Slider(
            0,
            1,
            value=0.3,
            step=0.01,
            label="Threshold",
            info="Lower the threshold to increase how many entities get predicted.",
            scale=1,
        )
        nested_ner = gr.Checkbox(
            value=examples[0][2],
            label="Nested NER",
            info="Allow for nested NER?",
            scale=0,
        )
    output = gr.HighlightedText(label="Predicted Entities")
    submit_btn = gr.Button("Submit")
    examples = gr.Examples(
        examples,
        fn=ner,
        inputs=[input_text, labels, threshold, nested_ner],
        outputs=output,
        cache_examples=True,
    )

    # Submitting
    input_text.submit(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    labels.submit(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    threshold.release(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    submit_btn.click(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    nested_ner.change(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )

demo.queue()
demo.launch(debug=True,share=True)