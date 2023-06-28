# Transformer
I want to learn how transformers work to better understand language models. So I'm going to try and build one. I began by learning pytorch just the basics. Once that was good, I moved on to construct the model in pytorch. Next I migrated the model to Google Collab so I can run it on the GPUs there and test it out! You can take a look, there will be more detailed info as well as setup + running it all over there. <br><br>
Link: [ coming soon ] <br><br>
## My Process
### Learning Tranformers
I began by learning how the transformer works at a high level, and understanding the math and structure. <br><br> The challenge here was understanding the math. What is attention and how does it work? What are the encoders, decoders, multiple attention heads, etc. doing intuively? Those kind of questions took me a while to sufficiently understand and I am in fact still understanding them.
### Building the Model in PyTorch
I then followed a tutorial to actually code it in pytorch as I am new to this all. I made sure to understand every line myself and work the math and computations out myself in a playground before programming. I also didn't copy the code they wrote, I just originally used it to get an idea of the model's organization. Once I had a clear idea of the nn.module nesting I wrote the classes myself, checking in with the tutorial to make sure I was on the right track. I know my code is quite similar...and that's okay. It wasn't copied, it was used to learn because ultimately this project is for learning -- not for showing that I did something new. <br><br> The challenges at this stage for me was understanding how the architecture translates into code. Things like understanding how we represent each component, how to use pytorch and what those modules mean, how the backpropogation will work, how the forward propogation will work. I understood that quickly after coding the `Attention` class so afterwards it was pretty simple.
### Training and Testing
[ coming soon ]
## Sources
[ coming soon ]
