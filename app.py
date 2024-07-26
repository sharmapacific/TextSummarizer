import gradio as gr
from transformers import pipeline
import logging


class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.translator_en = None
        self.translator_back = None        
        self.model_name = model_name
        self.summarizer = pipeline("summarization", model=model_name)

    def specify_language(self, source_lang, target_lang):
        if source_lang != 'English':
            self.translator_en = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{language_options[source_lang]}-en")
        else:
            self.translator_en = None

        if target_lang != 'English':
            self.translator_back = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{language_options[target_lang]}")
        else:
            self.translator_back = None

    def summarize(self, text, max_length=130, min_length=30):
        try:
            if self.translator_en:
                text = self.translator_en(text)[0]['translation_text']
            
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            
            if self.translator_back:
                summary = self.translator_back(summary)[0]['translation_text']
                
            return summary
        except Exception as e:
            logging.error(f"Error during summarize: {e}")
            return f"An error occurred: {e}"


# Initialize the summary app with the default model - facebook/bart-large-cnn
app = TextSummarizer()

def summarize_item(text, max_length, min_length, model_name, source_lang, target_lang):
    app.__init__(model_name)
    app.specify_language(source_lang, target_lang)
    summary = app.summarize(text, max_length, min_length)
    logging.info(f"Summary: {summary}")
    return text, summary

language_options = {
    "English": "en",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh",    
    "French": "fr",
    "Russian": "ru"
}

model_options = ["facebook/bart-large-cnn", "t5-base"]

# Define Gradio interface
iface = gr.Interface(
    fn=summarize_item,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter text here...", label="Text to Summarize"),
        gr.Slider(minimum=10, maximum=350, value=130, label="Maximum Summary Length"),
        gr.Slider(minimum=10, maximum=350, value=30, label="Minimum Summary Length"),
        gr.Dropdown(model_options, value="facebook/bart-large-cnn", label="Select Model"),
        gr.Dropdown(list(language_options.keys()), value="English", label="Select Input Language"),
        gr.Dropdown(list(language_options.keys()), value="English", label="Select Output Language")
    ],
    outputs=[
        gr.Textbox(label="Input Text"),
        gr.Textbox(label="Summary Text")
    ],
    title="Text summary App",
    description="Text summary using multiple AI models. Adjust summary length and language. Compare original and summarized content side-by-side.",
)

iface.launch()
