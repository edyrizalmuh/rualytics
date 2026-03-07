---
title: "What is a P-Value?"
date: 2024-03-07
description: "A plain English explanation of p-values for beginners, with simple examples."
tags: ["statistics", "beginner", "hypothesis-testing"]
categories: ["Statistics"]
math: true
draft: false
translationKey: pvalue-post
---

## Introduction

A p-value is one of the most misunderstood concepts in statistics...

## The Simple Explanation

If we assume the null hypothesis is true, the p-value tells us
how likely we are to see results at least as extreme as ours.

## Example

Suppose we test whether a coin is fair. We flip it 100 times
and get 60 heads. Our p-value answers:

> "If the coin were truly fair, how likely is it to get
> 60 or more heads just by chance?"

## The Formula

The test statistic for this example:

$$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$$

## In Python
```python
from scipy import stats

observed_heads = 60
n_flips = 100
expected_prob = 0.5

stat, pvalue = stats.binom_test(observed_heads, n_flips, expected_prob)
print(f"P-value: {pvalue:.4f}")
```