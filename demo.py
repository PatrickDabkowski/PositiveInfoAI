import inference
import gradio as gr

def bot(bart_device = 'cpu', sd_device = 'mps'):
        '''returns callable Bot object that returns positive text and generated image 
        based on currently most positive among most popular Wikipedia articles'''
        return inference.Bot(bart_device, sd_device)

def fn(is_new):
        # gradio Interface input function
        # is_new parameter drops previous title
        global agent
        print(is_new)
        text, img = agent.generate(is_new)
        return img.images[0], text


if __name__ == "__main__":
        agent = bot()
        demo = gr.Interface(fn, inputs=[gr.Radio([True, False])], outputs=[gr.Image(label="Graphics"), gr.Textbox(label="Post")])
        demo.launch(share=True)