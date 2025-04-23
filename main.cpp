#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_render.h>
#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include <string.h>

using namespace std;
const int maxIter = 500;

float MandelBrot(complex<double> c) {
  int i = 0;
  complex<double> z(0, 0);
  while (i <= maxIter && norm(z) <= 4) {
    z = pow(z, 2) + c;
    i++;
  }
  float mod = sqrt(norm(z));
  float itet = float(i) - log2(max(1.0f, log2(mod)));
  return itet;
}

float Julia(complex<double> c, complex<double> z) {
  int i = 0;
  while (i <= maxIter && norm(z) <= 4) {
    z = pow(z, 2) + c;
    i++;
  }
  float mod = sqrt(norm(z));
  float itet = float(i) - log2(max(1.0f, log2(mod)));
  return itet;
}

int main() {

  // Generation of MandelBrot set
  int pixelSize = 1;
  float a = -1;
  float b = 1;
  float step = 0.0005;
  int W = 1000;
  int H = 1000;
  int w = (b - a) / step;
  int h = w;
  float scale = 1.0;
  uint32_t *pixels = new uint32_t[w * h];

  complex<double> c(-0.7, 0.27015);

  for (int i = 0; i < h; i++) {
    float y = a + i * step;
    for (int j = 0; j < w; j++) {
      float x = a + j * step;
      complex<float> z(x, y);
      // float iter = Julia(c, z);
      float iter = MandelBrot(z);
      // float norm = Julia(c, z) / float(maxIter); // Linear Scaling
      float norm = pow(iter / float(maxIter), 0.6f); // Gamm Scaling
      int grad = int(norm * 255.0f);
      // cout << j + i * w << " " << i << " " << j << endl;
      pixels[j + i * w] =
          (0xFF) | (int(grad) << 8) | (int(grad) << 16) | (int(grad) << 24);
    }
  }

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
                                           SDL_TEXTUREACCESS_STREAMING, w, h);

  // SDL_Rect pixel = {0, 0, 1, 1};
  bool quit = false;
  // bool drawn = false;

  SDL_UpdateTexture(texture, NULL, pixels, w * sizeof(Uint32));
  // SDL_Rect src_Rect{1500, 1500, 1000, 1000};

  SDL_Rect src_Rect{int((w / 2) - scale * w / 2), int((h / 2) - scale * h / 2),
                    w, h};
  SDL_Rect dest_Rect{0, 0, W, H};

  cout << "Done" << endl;
  int x_offset = 0;
  int y_offset = 0;
  int panSpeed = 5;

  while (!quit) {

    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_QUIT) {
        quit = true;
      } else if (ev.type == SDL_KEYDOWN) {
        switch (ev.key.keysym.sym) {
        case SDLK_i: {
          cout << "in";
          scale -= 0.005;
          scale = scale < 0 ? 0 : scale;
          break;
        }
        case SDLK_o: {
          scale += 0.1;
          scale = scale > 1.0 ? 1.0 : scale;
          break;
        }
        case SDLK_LEFT: {
          x_offset -= panSpeed;
          break;
        }
        case SDLK_RIGHT: {
          x_offset += panSpeed;
          break;
        }
        case SDLK_UP: {
          y_offset -= panSpeed;
          break;
        }
        case SDLK_DOWN: {
          y_offset += panSpeed;
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
        case SDLK_q: {
          quit = true;
          break;
        }
        }
      }
    }

    int cx = int((w / 2) - scale * w / 2);
    int cy = int((h / 2) - scale * h / 2);
    src_Rect.x = cx;
    src_Rect.y = cy;
    src_Rect.w = int(w * scale);
    src_Rect.h = int(h * scale);

    // Apply panning
    src_Rect.x -= x_offset;
    src_Rect.y -= y_offset;

    // Ensure that the source rectangle doesn't exceed the image's bounds
    if (src_Rect.x < 0) {
      src_Rect.x = 0;
    } else if (src_Rect.x > (w - src_Rect.w)) {
      src_Rect.x = w - src_Rect.w;
    }

    if (src_Rect.y < 0) {
      src_Rect.y = 0;
    } else if (src_Rect.y > (h - src_Rect.h)) {
      src_Rect.y = h - src_Rect.h;
    }

    // Make sure the offsets are relative to the center of the rectangle
    x_offset = cx - src_Rect.x;
    y_offset = cy - src_Rect.y;

    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, &src_Rect, &dest_Rect);
    SDL_RenderPresent(renderer);
  }

  SDL_Quit();

  return 0;
}
