import random
import colorsys


def generate_unique_color(existing_colors):
    # Generate random hue value between 0 and 360
    hue = random.randint(0, 360)

    def hue_distance(h1, h2):
        # Calculate the shortest distance between two hue values
        distance = abs(h1 - h2)
        return min(distance, 360 - distance)

    def find_furthest_color(hue):
        # Find the color in existing_colors that is furthest from the given hue
        furthest_distance = 0
        furthest_color = None

        for color in existing_colors:
            color_hue = colorsys.rgb_to_hls(*color)[0] * 360
            distance = hue_distance(hue, color_hue)

            if distance > furthest_distance:
                furthest_distance = distance
                furthest_color = color

        return furthest_color

    def is_contrast_sufficient(color_param):
        # Check if the generated color has sufficient contrast against white
        return abs(color_param[0] - 255) >= 64 or abs(color_param[1] - 255) >= 64 or abs(color_param[2] - 255) >= 64

    # Find the color in existing_colors that is furthest from the generated hue
    furthest_color = find_furthest_color(hue)

    # Loop until a color with sufficient contrast is generated
    max_tries = 100
    while True:
        # Set saturation to a random value between 40 and 100 (to avoid very pale colors)
        saturation = random.randint(40, 100)

        # Set lightness to a random value between 30 and 70 (to avoid very dark or very light colors)
        lightness = random.randint(30, 70)

        # Convert HSL values to RGB values
        rgb = colorsys.hls_to_rgb(hue / 360.0, lightness / 100.0, saturation / 100.0)
        color = tuple(int(component * 255) for component in rgb)

        # Ensure good contrast against white
        if is_contrast_sufficient(color) or max_tries == 0:
            return color

        # Adjust the generated hue based on the furthest color
        hue_difference = hue_distance(hue, furthest_color[0] * 360)
        hue_shift = 30 + random.randint(0, 30)
        hue_shift *= -1 if hue_difference < 180 else 1
        hue = (hue + hue_shift) % 360

        max_tries -= 1
