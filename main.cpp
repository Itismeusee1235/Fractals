#include "fractal.cuh"
#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_render.h>
#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include <string.h>

using namespace std;
const int maxIter = 2000;

int main() {

  // Generation of MandelBrot set
  int pixelSize = 1;
  int W = 1000;
  int H = 1000;

  double c_x = 0;
  double c_y = 0;
  double w = 2;
  double h = 2;
  double step = 0.001;
  int set = 0;
  uint32_t *pixels = new uint32_t[W * H];

  SDL_Window *window;
  SDL_Renderer *renderer;
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    printf("Failed to init, %s ", SDL_GetError());
    return -1;
  }

  window = SDL_CreateWindow("Fractals", SDL_WINDOWPOS_UNDEFINED,
                            SDL_WINDOWPOS_UNDEFINED, W, H, SDL_WINDOW_SHOWN);
  if (!window) {
    printf("Failed to make window, %s, ", SDL_GetError());
  }

  renderer = SDL_CreateRenderer(window, -1, 0);
  if (!renderer) {
    printf("Failed to make renderer, %s, ", SDL_GetError());
  }

  SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                                           SDL_TEXTUREACCESS_STREAMING, W, H);

  SDL_Rect src_Rect{0, 0, W, H};
  SDL_Rect dest_Rect{0, 0, W, H};

  int x_offset = 0;
  int y_offset = 0;
  int panSpeed = 5;

  bool draw = true;

  bool quit = false;
  int m_x = 0;
  int m_y = 0;

  while (!quit) {

    SDL_Event ev;
    SDL_GetMouseState(&m_x, &m_y);

    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_QUIT) {
        quit = true;
      } else if (ev.type == SDL_KEYDOWN) {
        switch (ev.key.keysym.sym) {
        case SDLK_i: {
          step = 99.001 * step / 100;
          draw = true;
          break;
        }
        case SDLK_o: {
          step = 100.999 * step / 100;
          draw = true;
          break;
        }
        case SDLK_LEFT: {
          c_x -= step * 10;
          draw = true;
          break;
        }
        case SDLK_RIGHT: {
          c_x += step * 10;
          draw = true;
          break;
        }
        case SDLK_UP: {
          c_y += step * 10;
          draw = true;
          break;
        }
        case SDLK_DOWN: {
          c_y -= step * 10;
          draw = true;
          break;
        }
        case SDLK_f: {
          panSpeed++;
          break;
        }
        case SDLK_s: {
          panSpeed--;
          break;
        }
        case SDLK_TAB: {
          set = (set + 1) % 3;
          c_x = 0;
          c_y = 0;
          step = 0.001;
          draw = true;
          break;
        }
        case SDLK_q: {
          quit = true;
          break;
        }
        }
      } else if (ev.type == SDL_MOUSEWHEEL) {
        // Get mouse position in world coordinates before zoom
        double world_x_before = c_x + (m_x - W / 2.0) * step;
        double world_y_before = c_y + (H / 2.0 - m_y) * step;

        // Adjust step (zoom)
        double zoomFactor = 1.0 - ev.wheel.y * 0.1;
        if (zoomFactor > 0.0)
          step *= zoomFactor;

        // Get mouse position in world coordinates after zoom
        double world_x_after = c_x + (m_x - W / 2.0) * step;
        double world_y_after = c_y + (H / 2.0 - m_y) * step;

        // Shift view so that the point under the cursor stays under the
        // cursor
        c_x += world_x_before - world_x_after;
        c_y += world_y_before - world_y_after;

        draw = true;
      } else if (ev.type == SDL_MOUSEBUTTONDOWN) {
      }
    }

    if (draw) {
      runFractal(pixels, maxIter, W, H, c_x, c_y, step, set);
      SDL_UpdateTexture(texture, NULL, pixels, W * sizeof(Uint32));
      draw = false;
    }

    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, &src_Rect, &dest_Rect);
    SDL_RenderPresent(renderer);
  }

  SDL_Quit();

  return 0;
}
