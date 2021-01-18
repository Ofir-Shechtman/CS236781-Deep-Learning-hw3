r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=200,
        seq_len=50,
        h_dim=256,
        n_layers=3,
        dropout=0.1,
        learn_rate=0.001,
        lr_sched_factor=0.9,
        lr_sched_patience=4,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


def part1_generation_params():
    start_seq = "ACT I. SCENE 1.\n" \
                "Rousillon. The COUNT'S palace\n" \
                "Enter BERTRAM, the COUNTESS OF ROUSILLON, HELENA, and LAFEU, all in black\n" \
                "COUNTESS. In delivering my son from me, I bury a second husband.\n" \
                "BERTRAM. And I in going, madam, weep o'er my father's death anew;\n" \
                "but I must attend his Majesty's command, to whom I am now in\n" \
                "ward, evermore in subjection.\n" \
                "LAFEU. You shall find of the King a husband, madam; you, sir, a father\n." \
                "He that so generally is at all times good must of\n"
    temperature = 0.5
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======

    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

If we wanted to train on the whole text, we would need to load a matrix with (T,V) dimensions, where T is the size of
the whole text. It would be a problem to load this matrix into the memory. Further more, if we leave the hidden state
the same size, it wouldn't be effective, because we would try to use a relativly small matrix to remember the context
of the whole text. Also, we would argue that the context should be of limited size, because the connections between
the start of the text and the middle of it is similar to the connection between 2 random texts.

"""

part1_q2 = r"""
**Your answer:**

As we generating the text we send the previous generated text to the model, so the model learns the generated text as we
are generating it. Meaning it remembers not only the original sequence, but also the text it already generated. 

"""

part1_q3 = r"""
**Your answer:**

The text continues along the batches, so shuffling the batches would cause the model to learn a permutation of the text
and not the text itself.

"""

part1_q4 = r"""
**Your answer:**

**ANSWER PART 1**

2. using a very high temperature will cause the distribution to be very close to uniform - meaning the probability to draw any sample will be 
close to equal for each sample. this is not what we want for sampling, because we want the characters which had higher scores to have a higher probability of being samples than the 
others.

3. using a very low distribution will cause the distribution to be less uniform, meaning the probability for each char to be sampled will be higher
the higher its score was. this is usefull in the case the class scores were originally very close together - by diving them by a low T,
the differnces will be more noticeable, causing the softmax function to create higher probabilities for the characters with higher score to be drawn.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=53, h_dim=256, z_dim=128, x_sigma2=0.001, learn_rate=3e-4, betas=(0.9, 0.999),)
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None

def part3_gan_hyperparams():
    hypers = dict(
        batch_size=53,
        z_dim=100,
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=2e-4,
            betas=(0.5, 0.999),
            #weight_decay=2e-3
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=2e-4,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
