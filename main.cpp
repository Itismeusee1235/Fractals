#include "SDL_GUI.h"
#include "fractal.cuh"
#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include <string.h>

using namespace std;
const int maxIter = 2000;

int main() {

  // Generation of MandelBrot set
  int pixelSize = 1;
  int W = 700;
  int H = 700;

  double c_x = 0;
  double c_y = 0;
  double w = 2;
  double h = 2;
  double step = 0.001;
  int set = 0;
  uint32_t *pixels = new uint32_t[W * H];

  bool quit = false;
  bool draw = true;

  GUI gui;
  gui.GUI_Initialize("Fractals", W, H);

  while (!quit) {

    gui.GUI_eventHandler(quit, draw, c_x, c_y, step, set);
    if (draw) {
      runFractal(pixels, maxIter, W, H, c_x, c_y, step, set);
      gui.GUI_WriteTexture(pixels);
      draw = false;
    }

    gui.GUI_render();
  }

  SDL_Quit();

  return 0;
}
