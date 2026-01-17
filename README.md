![Training Status](https://github.com/Spinachboul/nn_simulate/tree/main/.github/workflows/train.yml/badge.svg)

# Neural Network Intuition Simulator

This repository is a **learning game**.

You do not train neural networks here.
You **design** them and observe how your decisions affect learning.

---

## How to Play

1. Open `model.yaml`
2. Change one or more parameters
3. Commit your changes
4. Open **GitHub Actions**
5. Read why your model failed or succeeded
6. Fix your design and try again

---

## Parameters You Control

- task
- activation
- loss
- normalization
- depth

Some combinations **fail instantly**.
Some train slowly.
Some overfit.
Some work well.

Your job is to learn *why*.

---

## Learning Philosophy

- No real datasets
- No gradients
- No randomness

This simulator encodes common deep learning failure modes and best practices
to give **fast, causal feedback**.

---

## Tip

Start by changing **only one parameter at a time**.
