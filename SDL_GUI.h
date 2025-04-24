#pragma once
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL_render.h>
#include <iostream>

class GUI {
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *texture;
  SDL_Rect rect;
  int width;
  int height;

public:
  bool GUI_Initialize(char *name, int width, int height);
  void GUI_WriteTexture(uint32_t *pixels);
  void GUI_render();
  void GUI_eventHandler(bool &quit, bool &draw, double &c_x, double &c_y,
                        double &step, int &set);
  ~GUI();
};
