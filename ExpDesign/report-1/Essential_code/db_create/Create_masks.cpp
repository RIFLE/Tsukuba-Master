#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void process_image_pair(const fs::path& original_image_path, const fs::path& modified_image_path, const fs::path& output_directory)
{
    Mat original_image = imread(original_image_path.string(), IMREAD_GRAYSCALE);
    Mat modified_image = imread(modified_image_path.string(), IMREAD_GRAYSCALE);

    if (original_image.empty() || modified_image.empty())
    {
        cout << "Could not open or find one of the images: " << original_image_path << " or " << modified_image_path << endl;
        return;
    }

    if (original_image.size() != modified_image.size())
    {
        cout << "The two images have different sizes: " << original_image_path << " and " << modified_image_path << endl;
        return;
    }

    Mat mask_image = Mat::zeros(original_image.size(), CV_8UC1);

    for (int y = 0; y < original_image.rows; y++)
    {
        for (int x = 0; x < original_image.cols; x++)
        {
            if (original_image.at<uchar>(y, x) != modified_image.at<uchar>(y, x))
            {
                mask_image.at<uchar>(y, x) = 255;
            }
        }
    }

    fs::path output_mask_path = output_directory / original_image_path.filename();
    imwrite(output_mask_path.string(), mask_image);
    cout << "Mask image saved as " << output_mask_path << endl;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        cout << "Usage: create_mask <original_images_directory> <modified_images_directory> <output_directory>" << endl;
        return -1;
    }

    fs::path original_images_directory(argv[1]);
    fs::path modified_images_directory(argv[2]);
    fs::path output_directory(argv[3]);

    if (!fs::exists(original_images_directory) || !fs::exists(modified_images_directory) || !fs::exists(output_directory))
    {
        cout << "One of the specified directories does not exist." << endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(original_images_directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".pgm")
        {
            fs::path modified_image_path = modified_images_directory / entry.path().filename();
            if (fs::exists(modified_image_path))
            {
                process_image_pair(entry.path(), modified_image_path, output_directory);
            }
            else
            {
                cout << "Modified image not found for: " << entry.path().filename() << endl;
            }
        }
    }

    return 0;
}