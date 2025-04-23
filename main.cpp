#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_render.h>
#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include <string.h>

using namespace std;
const int maxIter = 50;

void saveAsPNG(const char *name, SDL_Renderer *renderer, SDL_Texture *texture) {

  SDL_Texture *target = SDL_GetRenderTarget(renderer);
  SDL_SetRenderTarget(renderer, texture);
  int width, height;
  SDL_QueryTexture(texture, NULL, NULL, &width, &height);
  SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
  SDL_RenderReadPixels(renderer, NULL, surface->format->format, surface->pixels,
                       surface->pitch);
  IMG_SavePNG(surface, name);
  SDL_FreeSurface(surface);
  SDL_SetRenderTarget(renderer, target);
  cout << "Saved" << endl;
}

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
  int a = -2;
  int b = 2;
  float step = 0.001;
  int W = 1000;
  int H = 1000;
  int w = (b - a) / 0.001;
  int h = w;
  float scale = 1.0;
  uint32_t *pixels = new uint32_t[w * h];

  complex<double> c(-0.7, 0.27015);

  for (float i = a; i <= b; i += step) {
    for (float j = a; j <= b; j += step) {
      // cout << i << " " << j << endl;
      complex<double> z(i, j);
      float iter = Julia(c, z);
      int X = (i - a) / (b - a) * h;
      int Y = (j - a) / (b - a) * w;
      int grad = iter * 5.1;
      pixels[X + Y * w] =
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

  saveAsPNG("Julia.png", renderer, texture);
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
          scale -= 0.1;
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
    src_Rect.w = w * scale;
    src_Rect.h = h * scale;

    src_Rect.x -= x_offset;
    src_Rect.y -= y_offset;

    if (src_Rect.x < 0) {
      src_Rect.x = 0;
    } else if (src_Rect.x > (w - src_Rect.w)) {
      src_Rect.x = w - src_Rect.w;
    }
    if (src_Rect.y < 0) {
      src_Rect.y = 0;
    } else if (src_Rect.y > (h - src_Rect.y)) {
      src_Rect.h = w - src_Rect.h;
    }

    x_offset = cx - src_Rect.x;
    y_offset = cy - src_Rect.y;

    cout << src_Rect.x << " " << src_Rect.y << endl;

    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, &src_Rect, &dest_Rect);
    SDL_RenderPresent(renderer);
  }

  SDL_Quit();

  return 0;
}
