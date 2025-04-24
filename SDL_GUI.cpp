#include "SDL_GUI.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL_render.h>
#include <iostream>
#include <stdint.h>

bool GUI::GUI_Initialize(char *name, int width, int height) {
  printf("Initializing\n");
  GUI::width = width;
  GUI::height = height;
  rect = SDL_Rect{0, 0, width, height};
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    printf("Failed to init SDL, %s, ", SDL_GetError());
    return false;
  }

  window =
      SDL_CreateWindow(name, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                       width, height, SDL_WINDOW_SHOWN);
  if (window == NULL) {
    printf("Failed to make window, %s, ", SDL_GetError());
    return false;
  }

  renderer = SDL_CreateRenderer(window, -1, 0);
  if (renderer == NULL) {
    printf("Failed to make renderer, %s, ", SDL_GetError());
  }

  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                              SDL_TEXTUREACCESS_STREAMING, width, height);
  if (texture == NULL) {
    printf("Failed to create texture, %s ,", SDL_GetError());
  }
  printf("Done initia;ize:");
  return true;
}

void GUI::GUI_render() {
  printf("Rendering\n");
  SDL_SetRenderDrawColor(renderer, 0x0, 0x0, 0x0, 0xFF);
  SDL_RenderCopy(renderer, texture, &rect, &rect);
  SDL_RenderPresent(renderer);
}

void GUI::GUI_WriteTexture(uint32_t *pixels) {
  printf("Writing texture\n");
  SDL_UpdateTexture(texture, &rect, pixels, sizeof(uint32_t) * width);
  printf("Done Writing\n");
}

void GUI::GUI_eventHandler(bool &quit, bool &draw, double &c_x, double &c_y,
                           double &step, int &set) {
  SDL_Event ev;
  int m_x, m_y;
  SDL_GetMouseState(&m_x, &m_y);
  printf("Handling event\n");

  while (SDL_PollEvent(&ev)) {
    if (ev.type == SDL_QUIT) {
      quit = true;
      return;
    } else if (ev.type == SDL_MOUSEWHEEL) {
      double x_b = c_x + (m_x - width / 2.0) * step;
      double y_b = c_y + (width / 2.0 - m_y) * step;

      double zoom = 1.0 - ev.wheel.y * 0.1;
      if (zoom > 0.0) {
        step *= zoom;
      }

      double x_a = c_x + (m_x - width / 2.0) * step;
      double y_a = c_y + (width / 2.0 - m_y) * step;

      c_x += x_b - x_a;
      c_y += y_b - y_a;

      draw = true;
    } else if (ev.type == SDL_KEYDOWN) {
      if (ev.key.keysym.sym == SDLK_q) {
        quit = true;
        break;
      }

      if (ev.key.keysym.sym == SDLK_TAB) {
        set = (set + 1) % 5;
        c_x = 0;
        c_y = 0;
        step = 0.001;
        draw = true;
      }
    }
  }
}

GUI::~GUI() {
  SDL_DestroyWindow(window);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyTexture(texture);
  SDL_Quit();
}
