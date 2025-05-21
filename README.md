# Balanced Rate-Distortion Optimization in Learned Image Compression

Unofficial Pytorch implementation of **CVPR2025** paper **Balanced Rate-Distortion Optimization in Learned Image Compression**.

Code offered by ChatGPT4.1, neither tested nor modified yet.

Training in progress, results will be released once finished.  

---
### Implement Detail

`python 3.9`, `torch 1.11`

---

### How to use

`trainTJ.py`: file that implements with Trajectory Optimization

`trianQP.py`: file that implements with Quadratic Programming

---
### Log
**2025/05/21:** Trajectory Optimization results go crazy and Quadratic Programming doesn't work as expected.

**2025/05/20:** Seems the aforementioned problem still not solved, and experiment results are very strange, might need the official code.

**2025/05/19:** ~~The Quadratic Programming works well, but the model with Trajectory Optimization goes worse.~~ **Fixed**


---

### RelatedLink

- Paper Link: https://arxiv.org/abs/2502.20161
- Author Github Lnk: https://github.com/1chizhang