#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// ------------------------ Утилиты ------------------------

namespace util {

std::string Trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string ToLower(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string Normalize(const std::string& value) {
    return ToLower(Trim(value));
}

std::optional<std::string> FindObjectKeyCaseInsensitive(const json& object, const std::string& input) {
    if (!object.is_object()) {
        return std::nullopt;
    }

    const std::string normalized = Normalize(input);
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (Normalize(it.key()) == normalized) {
            return it.key();
        }
    }

    return std::nullopt;
}

std::vector<int> JsonToIntVector(const json& value) {
    std::vector<int> result;

    if (!value.is_array()) {
        return result;
    }

    for (const auto& item : value) {
        if (item.is_number_integer()) {
            result.push_back(item.get<int>());
        } else if (item.is_string()) {
            try {
                result.push_back(std::stoi(item.get<std::string>()));
            } catch (...) {
            }
        }
    }

    return result;
}

std::vector<std::string> JsonToStringVector(const json& value) {
    std::vector<std::string> result;

    if (!value.is_array()) {
        return result;
    }

    for (const auto& item : value) {
        if (item.is_string()) {
            result.push_back(item.get<std::string>());
        } else {
            result.push_back(item.dump());
        }
    }

    return result;
}

std::vector<int> NormalizeIntVector(std::vector<int> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::vector<int> IntersectIntVectors(std::vector<int> left, std::vector<int> right) {
    left = NormalizeIntVector(std::move(left));
    right = NormalizeIntVector(std::move(right));

    std::vector<int> result;
    std::set_intersection(
        left.begin(), left.end(),
        right.begin(), right.end(),
        std::back_inserter(result)
    );

    return result;
}

std::vector<int> MergeUniqueIntVectors(std::vector<int> left, const std::vector<int>& right) {
    left.insert(left.end(), right.begin(), right.end());
    return NormalizeIntVector(std::move(left));
}

std::string JoinInts(const std::vector<int>& values) {
    if (values.empty()) {
        return "[]";
    }

    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    oss << ']';
    return oss.str();
}

std::string JoinStrings(const std::vector<std::string>& values) {
    if (values.empty()) {
        return "";
    }

    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    return oss.str();
}

}  // namespace util

// ------------------------ Результирующие структуры ------------------------

struct TopPrediction {
    std::string classId;
    float probability = 0.0f;
};

struct DialogContext {
    std::optional<std::string> enteredRank;
};

struct ImageRecognitionResult {
    std::string imagePath;
    std::string recognizedObjectId;
    std::string selectedWhere;

    std::vector<int> yearsFromButtons;
    std::vector<int> yearsFromRecognition;
    std::vector<int> finalYears;

    std::string size;
    std::string description;
    std::vector<std::string> possibleMilitaryRanks;

    std::vector<TopPrediction> topPredictions;
};

struct DecisionResult {
    bool recognitionUsed = false;

    std::optional<std::string> enteredRank;

    std::vector<int> yearsFromButtons;
    std::vector<ImageRecognitionResult> recognizedImages;

    std::vector<int> finalYears;
};

// ------------------------ Интерфейс ввода/вывода ------------------------

class ITextUI {
public:
    virtual ~ITextUI() = default;
    virtual void PrintLine(const std::string& text) = 0;
    virtual std::string ReadLine(const std::string& prompt) = 0;
};

class ConsoleUI final : public ITextUI {
public:
    void PrintLine(const std::string& text) override {
        std::cout << text << '\n';
    }

    std::string ReadLine(const std::string& prompt) override {
        std::cout << prompt;
        std::string line;
        std::getline(std::cin, line);
        return line;
    }
};

// ------------------------ Загрузка JSON ------------------------

class JsonFileLoader {
public:
    static json LoadFromFile(const std::string& path) {
        std::ifstream input(path);
        if (!input.is_open()) {
            throw std::runtime_error("Не удалось открыть файл: " + path);
        }

        json data;
        input >> data;
        return data;
    }
};

// ------------------------ Каталог распознавания ------------------------

class RecognitionCatalog {
public:
    explicit RecognitionCatalog(json catalog)
        : catalog_(std::move(catalog)) {
        if (!catalog_.is_object()) {
            throw std::runtime_error("Каталог распознавания должен быть JSON-объектом.");
        }
    }

    bool HasId(const std::string& id) const {
        return catalog_.contains(id);
    }

    const json& GetById(const std::string& id) const {
        if (!HasId(id)) {
            throw std::runtime_error("Идентификатор не найден в каталоге: " + id);
        }
        return catalog_.at(id);
    }

private:
    json catalog_;
};

// ------------------------ Разрешение ответов ------------------------

class AnswerResolver {
public:
    struct ResolveResult {
        std::optional<std::string> resolvedKey;
        std::optional<std::string> enteredRank;
    };

    static ResolveResult ResolveKey(const json& answers, const std::string& userInput) {
        ResolveResult result;

        if (!answers.is_object()) {
            return result;
        }

        const std::string trimmed = util::Trim(userInput);
        const std::string normalized = util::Normalize(userInput);

        if (answers.contains(trimmed)) {
            result.resolvedKey = trimmed;
            return result;
        }

        if (auto exactCi = FindKeyCaseInsensitive(answers, normalized)) {
            result.resolvedKey = exactCi;
            return result;
        }

        if (normalized == "quit") {
            if (auto quitKey = FindKeyCaseInsensitive(answers, "quit")) {
                result.resolvedKey = quitKey;
                return result;
            }
        }

        if (normalized == "path") {
            if (auto pathKey = FindKeyCaseInsensitive(answers, "path")) {
                result.resolvedKey = pathKey;
                return result;
            }
        }

        if (!normalized.empty() && normalized != "quit" && normalized != "path") {
            if (auto rankKey = FindKeyCaseInsensitive(answers, "rank")) {
                result.resolvedKey = rankKey;
                result.enteredRank = trimmed;
                return result;
            }
        }

        return result;
    }

private:
    static std::optional<std::string> FindKeyCaseInsensitive(const json& answers, const std::string& normalizedNeedle) {
        for (auto it = answers.begin(); it != answers.end(); ++it) {
            if (util::Normalize(it.key()) == normalizedNeedle) {
                return it.key();
            }
        }
        return std::nullopt;
    }
};

// ------------------------ TFLite: метаданные и инференс ------------------------

struct Meta {
    std::vector<std::string> class_names;
    std::string base_model;
    std::string preprocess;
    int img_h = 224;
    int img_w = 224;
};

Meta LoadMeta(const std::string& jsonPath) {
    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        throw std::runtime_error("Не удалось открыть classes.json: " + jsonPath);
    }

    json j;
    in >> j;

    Meta meta;

    if (!j.contains("class_names") || !j.at("class_names").is_array()) {
        throw std::runtime_error("classes.json: отсутствует массив class_names");
    }

    for (const auto& item : j.at("class_names")) {
        if (item.is_string()) {
            meta.class_names.push_back(item.get<std::string>());
        } else if (item.is_number_integer()) {
            meta.class_names.push_back(std::to_string(item.get<int>()));
        } else {
            throw std::runtime_error("classes.json: элементы class_names должны быть строками или целыми числами");
        }
    }

    if (j.contains("base_model") && j.at("base_model").is_string()) {
        meta.base_model = j.at("base_model").get<std::string>();
    }

    if (j.contains("preprocess") && j.at("preprocess").is_string()) {
        meta.preprocess = j.at("preprocess").get<std::string>();
    }

    if (j.contains("img_height") && j.at("img_height").is_number_integer()) {
        meta.img_h = j.at("img_height").get<int>();
    }

    if (j.contains("img_width") && j.at("img_width").is_number_integer()) {
        meta.img_w = j.at("img_width").get<int>();
    }

    meta.base_model = util::ToLower(meta.base_model);
    meta.preprocess = util::ToLower(meta.preprocess);

    return meta;
}

cv::Mat ResizeWithPadBgr(const cv::Mat& imgBgr, int targetW, int targetH) {
    if (imgBgr.empty()) {
        throw std::runtime_error("Пустое изображение для ResizeWithPadBgr");
    }

    const double scale = std::min(
        static_cast<double>(targetW) / static_cast<double>(imgBgr.cols),
        static_cast<double>(targetH) / static_cast<double>(imgBgr.rows)
    );

    const int newW = std::max(1, static_cast<int>(std::round(imgBgr.cols * scale)));
    const int newH = std::max(1, static_cast<int>(std::round(imgBgr.rows * scale)));

    cv::Mat resized;
    cv::resize(imgBgr, resized, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);

    cv::Mat canvas(targetH, targetW, CV_8UC3, cv::Scalar(0, 0, 0));
    const int x = (targetW - newW) / 2;
    const int y = (targetH - newH) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, newW, newH)));

    return canvas;
}

static const float IMAGENET_MEAN_B = 103.939f;
static const float IMAGENET_MEAN_G = 116.779f;
static const float IMAGENET_MEAN_R = 123.680f;

void PreprocessVggResnetInplaceBgr(cv::Mat& bgr) {
    bgr.convertTo(bgr, CV_32FC3);

    std::vector<cv::Mat> channels(3);
    cv::split(bgr, channels);

    channels[0] = channels[0] - IMAGENET_MEAN_B;
    channels[1] = channels[1] - IMAGENET_MEAN_G;
    channels[2] = channels[2] - IMAGENET_MEAN_R;

    cv::merge(channels, bgr);
}

void PreprocessEfficientnetInplaceRgb(cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 127.5, -1.0);
    bgr = rgb;
}

enum class PrepKind {
    VGG16,
    RESNET50,
    EFFICIENTNET
};

PrepKind SelectPrep(const std::string& marker) {
    if (marker == "vgg16") {
        return PrepKind::VGG16;
    }
    if (marker == "resnet50") {
        return PrepKind::RESNET50;
    }
    if (marker == "efficientnet") {
        return PrepKind::EFFICIENTNET;
    }
    return PrepKind::VGG16;
}

std::unique_ptr<tflite::Interpreter> LoadTFLite(
    const std::string& modelPath,
    std::unique_ptr<tflite::FlatBufferModel>& modelHolder
) {
    modelHolder = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!modelHolder) {
        throw std::runtime_error("Не удалось загрузить .tflite модель: " + modelPath);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    tflite::InterpreterBuilder(*modelHolder, resolver)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Не удалось создать интерпретатор TFLite");
    }

    interpreter->SetNumThreads(2);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("AllocateTensors() failed");
    }

    return interpreter;
}

std::vector<std::pair<int, float>> TopK(const std::vector<float>& probs, int k) {
    std::vector<std::pair<int, float>> idxProb;
    idxProb.reserve(probs.size());

    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        idxProb.emplace_back(i, probs[i]);
    }

    if (idxProb.empty()) {
        return {};
    }

    if (k > static_cast<int>(idxProb.size())) {
        k = static_cast<int>(idxProb.size());
    }

    std::partial_sort(
        idxProb.begin(),
        idxProb.begin() + k,
        idxProb.end(),
        [](const auto& left, const auto& right) {
            return left.second > right.second;
        }
    );

    idxProb.resize(k);
    return idxProb;
}

struct Pred {
    std::string path;
    std::vector<float> probs;
    std::vector<std::pair<int, float>> top3;
};

Pred PredictOne(
    tflite::Interpreter* interpreter,
    const std::string& imagePath,
    const Meta& meta,
    PrepKind prepKind
) {
    cv::Mat imgBgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (imgBgr.empty()) {
        throw std::runtime_error("Не удалось прочитать изображение: " + imagePath);
    }

    cv::Mat padded = ResizeWithPadBgr(imgBgr, meta.img_w, meta.img_h);

    switch (prepKind) {
        case PrepKind::VGG16:
        case PrepKind::RESNET50:
            PreprocessVggResnetInplaceBgr(padded);
            break;
        case PrepKind::EFFICIENTNET:
            PreprocessEfficientnetInplaceRgb(padded);
            break;
        default:
            PreprocessVggResnetInplaceBgr(padded);
            break;
    }

    if (!padded.isContinuous()) {
        padded = padded.clone();
    }

    TfLiteTensor* inputTensor = interpreter->tensor(interpreter->inputs()[0]);
    if (inputTensor == nullptr) {
        throw std::runtime_error("Не удалось получить входной тензор");
    }

    if (inputTensor->type != kTfLiteFloat32) {
        throw std::runtime_error("Поддерживается только float32 входной тензор");
    }

    if (inputTensor->dims == nullptr || inputTensor->dims->size != 4) {
        throw std::runtime_error("Ожидается входной тензор формы [1, H, W, 3]");
    }

    const int batch = inputTensor->dims->data[0];
    const int height = inputTensor->dims->data[1];
    const int width = inputTensor->dims->data[2];
    const int channels = inputTensor->dims->data[3];

    if (batch != 1 || height != meta.img_h || width != meta.img_w || channels != 3) {
        throw std::runtime_error("Размер входного тензора модели не совпадает с метаданными classes.json");
    }

    float* input = interpreter->typed_input_tensor<float>(0);
    const std::size_t bytes = static_cast<std::size_t>(meta.img_w) *
                              static_cast<std::size_t>(meta.img_h) * 3 * sizeof(float);

    if (padded.type() != CV_32FC3) {
        throw std::runtime_error("Внутренняя ошибка: ожидался CV_32FC3 после препроцессинга");
    }

    std::memcpy(input, padded.data, bytes);

    if (interpreter->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Invoke() failed");
    }

    const int outIdx = interpreter->outputs()[0];
    TfLiteTensor* outTensor = interpreter->tensor(outIdx);
    if (outTensor == nullptr) {
        throw std::runtime_error("Не удалось получить выходной тензор");
    }

    if (outTensor->type != kTfLiteFloat32) {
        throw std::runtime_error("Поддерживается только float32 выходной тензор");
    }

    int numClasses = 0;
    if (outTensor->dims != nullptr) {
        if (outTensor->dims->size == 2 && outTensor->dims->data[0] == 1) {
            numClasses = outTensor->dims->data[1];
        } else if (outTensor->dims->size == 1) {
            numClasses = outTensor->dims->data[0];
        }
    }

    if (numClasses <= 0) {
        throw std::runtime_error("Неожиданная форма выходного тензора");
    }

    const float* out = interpreter->typed_output_tensor<float>(0);
    std::vector<float> probs(static_cast<std::size_t>(numClasses));

    for (int i = 0; i < numClasses; ++i) {
        probs[static_cast<std::size_t>(i)] = out[i];
    }

    Pred pred;
    pred.path = imagePath;
    pred.probs = std::move(probs);
    pred.top3 = TopK(pred.probs, 3);

    return pred;
}

std::string LabelByIndex(const Meta& meta, int idx) {
    if (idx >= 0 && idx < static_cast<int>(meta.class_names.size())) {
        return meta.class_names[static_cast<std::size_t>(idx)];
    }
    return "class_" + std::to_string(idx);
}

// ------------------------ Обёртка над распознаванием ------------------------

class ImageRecognizer {
public:
    ImageRecognizer(const std::string& modelPath, const std::string& classesJsonPath)
        : meta_(LoadMeta(classesJsonPath)) {
        const std::string marker = !meta_.preprocess.empty() ? meta_.preprocess : meta_.base_model;
        prepKind_ = SelectPrep(marker);
        interpreter_ = LoadTFLite(modelPath, modelHolder_);
    }

    Pred Predict(const std::string& imagePath) {
        return PredictOne(interpreter_.get(), imagePath, meta_, prepKind_);
    }

    const Meta& GetMeta() const {
        return meta_;
    }

    std::string GetTop1ClassId(const Pred& pred) const {
        if (pred.top3.empty()) {
            throw std::runtime_error("Модель не вернула ни одного класса.");
        }

        return LabelByIndex(meta_, pred.top3.front().first);
    }

private:
    Meta meta_;
    PrepKind prepKind_ = PrepKind::VGG16;
    std::unique_ptr<tflite::FlatBufferModel> modelHolder_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
};

// ------------------------ Сессия распознавания ------------------------

class RecognitionSession {
public:
    RecognitionSession(const RecognitionCatalog& catalog, ImageRecognizer& recognizer, ITextUI& ui)
        : catalog_(catalog),
          recognizer_(recognizer),
          ui_(ui) {}

    DecisionResult Run(
        const json& savedStatement,
        const std::string& questionText,
        const DialogContext& context
    ) {
        DecisionResult result;
        result.recognitionUsed = false;
        result.enteredRank = context.enteredRank;
        result.yearsFromButtons = util::JsonToIntVector(savedStatement);

        ui_.PrintLine(questionText);

        while (true) {
            const std::string input = util::Trim(ui_.ReadLine("> "));
            const std::string normalized = util::Normalize(input);

            if (normalized == "quit") {
                FinalizeDecisionResult(result);
                return result;
            }

            if (input.empty()) {
                ui_.PrintLine("Путь к файлу не должен быть пустым. Введите путь к изображению или QUIT.");
                continue;
            }

            try {
                const ImageRecognitionResult imageResult = RecognizeSingleImage(
                    savedStatement,
                    input,
                    context
                );

                result.recognitionUsed = true;
                result.recognizedImages.push_back(imageResult);

                ui_.PrintLine("Изображение обработано.");
                ui_.PrintLine("Локальный результат по этому изображению: " + util::JoinInts(imageResult.finalYears));
                ui_.PrintLine("Введите путь к следующему изображению или QUIT для завершения.");

            } catch (const std::exception& ex) {
                ui_.PrintLine(std::string("Ошибка распознавания: ") + ex.what());
                ui_.PrintLine("Попробуйте снова ввести корректный путь к изображению или QUIT.");
            }
        }
    }

private:
    ImageRecognitionResult RecognizeSingleImage(
        const json& savedStatement,
        const std::string& imagePath,
        const DialogContext& context
    ) {
        const Pred prediction = recognizer_.Predict(imagePath);
        const std::string classId = recognizer_.GetTop1ClassId(prediction);

        if (!catalog_.HasId(classId)) {
            throw std::runtime_error(
                "Класс '" + classId + "' отсутствует в recognition.json. "
                "Проверьте соответствие class_names и ключей recognition.json."
            );
        }

        const json& entry = catalog_.GetById(classId);
        return ProcessRecognitionResult(savedStatement, imagePath, prediction, classId, entry, context);
    }

    ImageRecognitionResult ProcessRecognitionResult(
        const json& savedStatement,
        const std::string& imagePath,
        const Pred& prediction,
        const std::string& classId,
        const json& entry,
        const DialogContext& context
    ) {
        (void)context;

        if (!entry.contains("where") || !entry.at("where").is_object()) {
            throw std::runtime_error("В recognition.json у объекта '" + classId + "' отсутствует корректный раздел where.");
        }

        const json& where = entry.at("where");

        if (where.size() == 1) {
            const auto it = where.begin();
            return BuildRecognitionResult(
                savedStatement,
                imagePath,
                prediction,
                classId,
                it.key(),
                it.value(),
                entry
            );
        }

        std::vector<std::string> options;
        for (auto it = where.begin(); it != where.end(); ++it) {
            options.push_back(it.key());
        }

        ui_.PrintLine(
            "У распознанного объекта несколько возможных мест обнаружения. "
            "Укажите одно из них: " + util::JoinStrings(options)
        );

        while (true) {
            const std::string userWhere = util::Trim(ui_.ReadLine("> "));
            const auto resolvedKey = util::FindObjectKeyCaseInsensitive(where, userWhere);

            if (!resolvedKey.has_value()) {
                ui_.PrintLine("Некорректный ввод. Доступные варианты: " + util::JoinStrings(options));
                continue;
            }

            return BuildRecognitionResult(
                savedStatement,
                imagePath,
                prediction,
                classId,
                *resolvedKey,
                where.at(*resolvedKey),
                entry
            );
        }
    }

    ImageRecognitionResult BuildRecognitionResult(
        const json& savedStatement,
        const std::string& imagePath,
        const Pred& prediction,
        const std::string& classId,
        const std::string& whereKey,
        const json& yearsJson,
        const json& entry
    ) {
        ImageRecognitionResult result;
        result.imagePath = imagePath;
        result.recognizedObjectId = classId;
        result.selectedWhere = whereKey;

        result.yearsFromButtons = util::JsonToIntVector(savedStatement);
        result.yearsFromRecognition = util::JsonToIntVector(yearsJson);

        result.finalYears = util::IntersectIntVectors(result.yearsFromButtons, result.yearsFromRecognition);
        if (result.finalYears.empty()) {
            result.finalYears = util::MergeUniqueIntVectors(result.yearsFromButtons, result.yearsFromRecognition);
        }

        result.size = ExtractStringField(entry, "size");
        result.description = ExtractStringField(entry, "description");
        result.possibleMilitaryRanks = ExtractStringVectorField(entry, "range");

        for (const auto& item : prediction.top3) {
            TopPrediction top;
            top.classId = LabelByIndex(recognizer_.GetMeta(), item.first);
            top.probability = item.second;
            result.topPredictions.push_back(top);
        }

        return result;
    }

    void FinalizeDecisionResult(DecisionResult& result) const {
        result.yearsFromButtons = util::NormalizeIntVector(result.yearsFromButtons);

        if (result.recognizedImages.empty()) {
            result.finalYears = result.yearsFromButtons;
            return;
        }

        std::vector<int> intersection = result.yearsFromButtons;
        std::vector<int> unionYears = result.yearsFromButtons;

        for (const auto& imageResult : result.recognizedImages) {
            intersection = util::IntersectIntVectors(intersection, imageResult.yearsFromRecognition);
            unionYears = util::MergeUniqueIntVectors(unionYears, imageResult.yearsFromRecognition);
        }

        if (!intersection.empty()) {
            result.finalYears = intersection;
        } else {
            result.finalYears = unionYears;
        }
    }

    std::string ExtractStringField(const json& entry, const std::string& key) const {
        if (!entry.contains(key)) {
            return "отсутствует";
        }

        const json& value = entry.at(key);

        if (value.is_string()) {
            return value.get<std::string>();
        }

        if (value.is_array()) {
            std::vector<std::string> parts;
            for (const auto& item : value) {
                if (item.is_string()) {
                    parts.push_back(item.get<std::string>());
                } else {
                    parts.push_back(item.dump());
                }
            }
            return util::JoinStrings(parts);
        }

        return value.dump();
    }

    std::vector<std::string> ExtractStringVectorField(const json& entry, const std::string& key) const {
        if (!entry.contains(key)) {
            return {};
        }

        return util::JsonToStringVector(entry.at(key));
    }

private:
    const RecognitionCatalog& catalog_;
    ImageRecognizer& recognizer_;
    ITextUI& ui_;
};

// ------------------------ Движок дерева решений ------------------------

class DecisionTreeEngine {
public:
    DecisionTreeEngine(json treeRoot, const RecognitionCatalog& catalog, ImageRecognizer& recognizer, ITextUI& ui)
        : treeRoot_(std::move(treeRoot)),
          recognitionSession_(catalog, recognizer, ui),
          ui_(ui) {
        ValidateRoot();
    }

    DecisionResult Run() {
        const json* currentNode = &treeRoot_;

        while (true) {
            EnsureNodeHasQuestionAndAnswers(*currentNode);

            if (IsRecognitionLeafNode(*currentNode)) {
                const json statement = ExtractStatement(*currentNode);
                const std::string questionText = currentNode->at("question").get<std::string>();
                return recognitionSession_.Run(statement, questionText, context_);
            }

            ui_.PrintLine(currentNode->at("question").get<std::string>());
            const std::string userInput = ui_.ReadLine("> ");

            const json& answers = currentNode->at("answers");
            const auto resolveResult = AnswerResolver::ResolveKey(answers, userInput);

            if (!resolveResult.resolvedKey.has_value()) {
                ui_.PrintLine("Некорректный ответ. Попробуйте еще раз.");
                continue;
            }

            if (resolveResult.enteredRank.has_value()) {
                context_.enteredRank = resolveResult.enteredRank;
            }

            const json& next = answers.at(*resolveResult.resolvedKey);

            if (next.is_object()) {
                currentNode = &next;
                continue;
            }

            if (next.is_string()) {
                return ExecuteCommand(next.get<std::string>(), *currentNode);
            }

            throw std::runtime_error(
                "Некорректный формат узла answers: значение должно быть объектом или строковой командой."
            );
        }
    }

private:
    void ValidateRoot() const {
        if (!treeRoot_.is_object()) {
            throw std::runtime_error("Корень дерева вопросов должен быть JSON-объектом.");
        }
        EnsureNodeHasQuestionAndAnswers(treeRoot_);
    }

    static void EnsureNodeHasQuestionAndAnswers(const json& node) {
        if (!node.contains("question") || !node.at("question").is_string()) {
            throw std::runtime_error("Узел дерева не содержит строковое поле 'question'.");
        }
        if (!node.contains("answers") || !node.at("answers").is_object()) {
            throw std::runtime_error("Узел дерева не содержит объект 'answers'.");
        }
    }

    DecisionResult ExecuteCommand(const std::string& command, const json& currentNode) {
        if (command == "RETURN_STATEMENT") {
            return BuildStatementOnlyResult(currentNode);
        }

        if (command == "RECOGNITION_STATEMENT") {
            const json statement = ExtractStatement(currentNode);
            const std::string questionText = currentNode.at("question").get<std::string>();
            return recognitionSession_.Run(statement, questionText, context_);
        }

        throw std::runtime_error("Неизвестная команда в answers: " + command);
    }

    static json ExtractStatement(const json& node) {
        if (node.contains("statement")) {
            return node.at("statement");
        }
        return json::array();
    }

    DecisionResult BuildStatementOnlyResult(const json& node) {
        DecisionResult result;
        result.recognitionUsed = false;
        result.enteredRank = context_.enteredRank;
        result.yearsFromButtons = util::JsonToIntVector(ExtractStatement(node));
        result.finalYears = util::NormalizeIntVector(result.yearsFromButtons);
        return result;
    }

    static bool IsRecognitionLeafNode(const json& node) {
        if (!node.is_object()) {
            return false;
        }

        if (!node.contains("question") || !node.at("question").is_string()) {
            return false;
        }

        if (!node.contains("answers") || !node.at("answers").is_object()) {
            return false;
        }

        const json& answers = node.at("answers");
        bool hasQuit = false;
        bool hasPath = false;

        for (auto it = answers.begin(); it != answers.end(); ++it) {
            const std::string key = util::Normalize(it.key());

            if (key == "quit" &&
                it.value().is_string() &&
                it.value().get<std::string>() == "RETURN_STATEMENT") {
                hasQuit = true;
            }

            if (key == "path" &&
                it.value().is_string() &&
                it.value().get<std::string>() == "RECOGNITION_STATEMENT") {
                hasPath = true;
            }
        }

        return hasQuit && hasPath;
    }

private:
    json treeRoot_;
    RecognitionSession recognitionSession_;
    ITextUI& ui_;
    DialogContext context_;
};

// ------------------------ Вывод результата ------------------------

void PrintDecisionResult(const DecisionResult& result, ITextUI& ui) {
    if (result.enteredRank.has_value()) {
        ui.PrintLine("Указанное пользователем звание: " + *result.enteredRank);
    }

    ui.PrintLine("Годы, полученные из информации о пуговицах: " + util::JoinInts(result.yearsFromButtons));

    if (result.recognitionUsed) {
        ui.PrintLine("Количество распознанных изображений: " + std::to_string(result.recognizedImages.size()));

        for (std::size_t i = 0; i < result.recognizedImages.size(); ++i) {
            const auto& image = result.recognizedImages[i];

            ui.PrintLine("----------------------------------------");
            ui.PrintLine("Изображение #" + std::to_string(i + 1));
            ui.PrintLine("Путь к изображению: " + image.imagePath);
            ui.PrintLine("Распознанный объект: " + image.recognizedObjectId);
            ui.PrintLine("Выбранное место обнаружения: " + image.selectedWhere);
            ui.PrintLine("Годы, полученные из информации о пуговицах: " + util::JoinInts(image.yearsFromButtons));
            ui.PrintLine("Годы, полученные из описания распознанного изображения: " + util::JoinInts(image.yearsFromRecognition));
            ui.PrintLine("Локальный результат по изображению: " + util::JoinInts(image.finalYears));
            ui.PrintLine("Размер распознанного предмета: " + image.size);
            ui.PrintLine("Описание распознанного предмета: " + image.description);
            ui.PrintLine("Возможные воинские звания для распознанного предмета: " + util::JoinStrings(image.possibleMilitaryRanks));

            // Если нужно вернуть вывод top-3, можно раскомментировать:
            // if (!image.topPredictions.empty()) {
            //     ui.PrintLine("Топ-3 наиболее вероятных класса:");
            //     for (std::size_t j = 0; j < image.topPredictions.size(); ++j) {
            //         std::ostringstream oss;
            //         oss << "  #" << (j + 1) << ": "
            //             << image.topPredictions[j].classId
            //             << " ("
            //             << std::fixed << std::setprecision(2)
            //             << image.topPredictions[j].probability * 100.0f
            //             << "%)";
            //         ui.PrintLine(oss.str());
            //     }
            // }
        }

        ui.PrintLine("----------------------------------------");
    }

    ui.PrintLine("ИТОГОВОЕ РЕШЕНИЕ ПО ВСЕМ ПУГОВИЦАМ И ИЗОБРАЖЕНИЯМ: " + util::JoinInts(result.finalYears));
}

// ------------------------ main ------------------------

int main(int argc, char* argv[]) {
    try {
        if (argc < 5) {
            std::cerr << "Использование:\n";
            std::cerr << "  " << argv[0]
                      << " <questions.json> <recognition.json> <model.tflite> <classes.json>\n";
            return 1;
        }

        const std::string questionsPath = argv[1];
        const std::string recognitionPath = argv[2];
        const std::string modelPath = argv[3];
        const std::string classesPath = argv[4];

        const json questionsTree = JsonFileLoader::LoadFromFile(questionsPath);
        const json recognitionJson = JsonFileLoader::LoadFromFile(recognitionPath);

        ConsoleUI ui;
        RecognitionCatalog catalog(recognitionJson);
        ImageRecognizer recognizer(modelPath, classesPath);
        DecisionTreeEngine engine(questionsTree, catalog, recognizer, ui);

        const DecisionResult result = engine.Run();
        PrintDecisionResult(result, ui);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << '\n';
        return 1;
    }
}