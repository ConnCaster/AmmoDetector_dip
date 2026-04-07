#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"

using json = nlohmann::json;

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

std::string JsonToCompactString(const json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    return value.dump();
}

std::string StatementToString(const json& statement) {
    if (statement.is_array()) {
        std::ostringstream oss;
        oss << '[';
        for (std::size_t i = 0; i < statement.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << statement[i];
        }
        oss << ']';
        return oss.str();
    }

    if (statement.is_null()) {
        return "[]";
    }

    return JsonToCompactString(statement);
}

}  // namespace util

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

// ------------------------ Форматирование вывода ------------------------

class OutputFormatter {
public:
    static std::string BuildStatementText(const json& statement) {
        return "Решение: " + util::StatementToString(statement);
    }

    static std::vector<std::string> BuildRecognitionInfo(const std::string& id, const json& entry) {
        std::vector<std::string> lines;
        lines.push_back("Распознанный объект: " + id);

        if (entry.contains("where")) {
            lines.push_back("where:");
            const json& where = entry.at("where");
            if (where.is_object()) {
                for (auto it = where.begin(); it != where.end(); ++it) {
                    lines.push_back("  " + it.key() + ": " + util::StatementToString(it.value()));
                }
            } else {
                lines.push_back("  " + util::JsonToCompactString(where));
            }
        }

        if (entry.contains("size")) {
            lines.push_back("size: " + FormatAny(entry.at("size")));
        }

        if (entry.contains("description")) {
            lines.push_back("description: " + FormatAny(entry.at("description")));
        }

        return lines;
    }

private:
    static std::string FormatAny(const json& value) {
        if (value.is_string()) {
            return value.get<std::string>();
        }

        if (value.is_array()) {
            std::ostringstream oss;
            oss << '[';
            for (std::size_t i = 0; i < value.size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << FormatAny(value[i]);
            }
            oss << ']';
            return oss.str();
        }

        if (value.is_object()) {
            return value.dump();
        }

        return value.dump();
    }
};

// ------------------------ Разрешение ответов ------------------------

class AnswerResolver {
public:
    static std::optional<std::string> ResolveKey(const json& answers, const std::string& userInput) {
        if (!answers.is_object()) {
            return std::nullopt;
        }

        const std::string trimmed = util::Trim(userInput);
        const std::string normalized = util::Normalize(userInput);

        // 1. Точное совпадение
        if (answers.contains(trimmed)) {
            return trimmed;
        }

        // 2. Поиск по ключам без учета регистра
        if (auto exactCi = FindKeyCaseInsensitive(answers, normalized)) {
            return exactCi;
        }

        // 3. Специальные ключи
        if (normalized == "quit") {
            if (auto quitKey = FindKeyCaseInsensitive(answers, "quit")) {
                return quitKey;
            }
        }

        if (normalized == "path") {
            if (auto pathKey = FindKeyCaseInsensitive(answers, "path")) {
                return pathKey;
            }
        }

        // 4. Любой непустой ввод можно интерпретировать как RANK, если такой ключ есть
        if (!normalized.empty() && normalized != "quit" && normalized != "path") {
            if (auto rankKey = FindKeyCaseInsensitive(answers, "rank")) {
                return rankKey;
            }
        }

        return std::nullopt;
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

// ------------------------ Сессия "распознавания" ------------------------

class RecognitionSession {
public:
    RecognitionSession(const RecognitionCatalog& catalog, ITextUI& ui)
        : catalog_(catalog), ui_(ui) {}

    void Run(const json& savedStatement, const std::string& questionText) const {
        ui_.PrintLine(questionText);

        while (true) {
            const std::string input = util::Trim(ui_.ReadLine("> "));
            const std::string normalized = util::Normalize(input);

            if (normalized == "quit") {
                ui_.PrintLine(OutputFormatter::BuildStatementText(savedStatement));
                return;
            }

            if (catalog_.HasId(input)) {
                ui_.PrintLine(OutputFormatter::BuildStatementText(savedStatement));

                const json& entry = catalog_.GetById(input);
                const auto lines = OutputFormatter::BuildRecognitionInfo(input, entry);
                for (const auto& line : lines) {
                    ui_.PrintLine(line);
                }
                return;
            }

            ui_.PrintLine("Некорректный ввод. Введите существующий номер из recognition.json или QUIT.");
        }
    }

private:
    const RecognitionCatalog& catalog_;
    ITextUI& ui_;
};

// ------------------------ Движок дерева решений ------------------------

class DecisionTreeEngine {
public:
    DecisionTreeEngine(json treeRoot, const RecognitionCatalog& catalog, ITextUI& ui)
        : treeRoot_(std::move(treeRoot)),
          recognitionSession_(catalog, ui),
          ui_(ui) {
        ValidateRoot();
    }

    void Run() {
        const json* currentNode = &treeRoot_;

        while (true) {
            EnsureNodeHasQuestionAndAnswers(*currentNode);

            // Специальный режим:
            // вопрос про "путь к файлу или QUIT",
            // но фактически принимаем QUIT или цифру-ключ из recognition.json
            if (IsRecognitionLeafNode(*currentNode)) {
                const json statement = ExtractStatement(*currentNode);
                const std::string questionText = currentNode->at("question").get<std::string>();
                recognitionSession_.Run(statement, questionText);
                return;
            }

            ui_.PrintLine(currentNode->at("question").get<std::string>());
            const std::string userInput = ui_.ReadLine("> ");

            const json& answers = currentNode->at("answers");
            const auto resolvedKey = AnswerResolver::ResolveKey(answers, userInput);

            if (!resolvedKey.has_value()) {
                ui_.PrintLine("Некорректный ответ. Попробуйте еще раз.");
                continue;
            }

            const json& next = answers.at(*resolvedKey);

            if (next.is_object()) {
                currentNode = &next;
                continue;
            }

            if (next.is_string()) {
                ExecuteCommand(next.get<std::string>(), *currentNode);
                return;
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

    void ExecuteCommand(const std::string& command, const json& currentNode) {
        if (command == "RETURN_STATEMENT") {
            PrintStatementOfNode(currentNode);
            return;
        }

        if (command == "RECOGNITION_STATEMENT") {
            const json statement = ExtractStatement(currentNode);
            const std::string questionText = currentNode.at("question").get<std::string>();
            recognitionSession_.Run(statement, questionText);
            return;
        }

        throw std::runtime_error("Неизвестная команда в answers: " + command);
    }

    static json ExtractStatement(const json& node) {
        if (node.contains("statement")) {
            return node.at("statement");
        }
        return json::array();
    }

    void PrintStatementOfNode(const json& node) {
        ui_.PrintLine(OutputFormatter::BuildStatementText(ExtractStatement(node)));
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

            if (key == "quit" && it.value().is_string() &&
                it.value().get<std::string>() == "RETURN_STATEMENT") {
                hasQuit = true;
                }

            if (key == "path" && it.value().is_string() &&
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
};

// ===================================================================

#include <filesystem>
#include <memory>
#include <algorithm>
#include <cstring>


#include <opencv2/opencv.hpp>

// JSON (header-only): https://github.com/nlohmann/json
#include "nlohmann/json.hpp"

// TFLite
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/c/common.h"

namespace fs = std::filesystem;

// -------------------- Утилиты --------------------
struct Meta {
    std::vector<std::string> class_names;
    std::string base_model;   // "vgg16" | "resnet50" | "efficientnet"
    std::string preprocess;   // тот же маркер, что и base_model
    int img_h = 224;
    int img_w = 224;
};

Meta load_meta(const std::string& json_path) {
    std::ifstream in(json_path);
    if (!in) throw std::runtime_error("Не удалось открыть classes.json: " + json_path);
    json j; in >> j;

    Meta m;
    if (!j.contains("class_names")) throw std::runtime_error("classes.json: нет поля class_names");
    for (auto& v : j["class_names"]) m.class_names.push_back(v.get<std::string>());

    if (j.contains("base_model"))  m.base_model  = j["base_model"].get<std::string>();
    if (j.contains("preprocess"))  m.preprocess  = j["preprocess"].get<std::string>();
    if (j.contains("img_height"))  m.img_h       = j["img_height"].get<int>();
    if (j.contains("img_width"))   m.img_w       = j["img_width"].get<int>();

    std::transform(m.preprocess.begin(), m.preprocess.end(), m.preprocess.begin(), ::tolower);
    std::transform(m.base_model.begin(), m.base_model.end(), m.base_model.begin(), ::tolower);
    return m;
}

std::vector<std::string> gather_images(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& p : fs::recursive_directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        auto ext = p.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            files.push_back(p.path().string());
        }
    }
    return files;
}

// -------------------- resize_with_pad --------------------
cv::Mat resize_with_pad_bgr(const cv::Mat& img_bgr, int target_w, int target_h) {
    double scale = std::min(
            static_cast<double>(target_w) / img_bgr.cols,
            static_cast<double>(target_h) / img_bgr.rows
    );
    int new_w = std::max(1, static_cast<int>(std::round(img_bgr.cols * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(img_bgr.rows * scale)));

    cv::Mat resized;
    cv::resize(img_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(0, 0, 0));
    int x = (target_w - new_w) / 2;
    int y = (target_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, new_w, new_h)));
    return canvas;
}

// -------------------- Препроцессинги --------------------
static const float IMAGENET_MEAN_B = 103.939f;
static const float IMAGENET_MEAN_G = 116.779f;
static const float IMAGENET_MEAN_R = 123.680f;

// VGG16/ResNet50 (caffe-подобный): вход BGR float32, вычитание средних по каналам
void preprocess_vgg_resnet_inplace_bgr(cv::Mat& bgr) {
    bgr.convertTo(bgr, CV_32FC3);
    std::vector<cv::Mat> ch(3);
    cv::split(bgr, ch);
    ch[0] = ch[0] - IMAGENET_MEAN_B;
    ch[1] = ch[1] - IMAGENET_MEAN_G;
    ch[2] = ch[2] - IMAGENET_MEAN_R;
    cv::merge(ch, bgr);
}

// EfficientNet: RGB float32 в диапазоне [-1, 1]
void preprocess_efficientnet_inplace_rgb(cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 127.5, -1.0); // x/127.5 - 1
    bgr = rgb;
}

enum class PrepKind { VGG16, RESNET50, EFFICIENTNET };

PrepKind select_prep(const std::string& marker) {
    if (marker == "vgg16") return PrepKind::VGG16;
    if (marker == "resnet50") return PrepKind::RESNET50;
    if (marker == "efficientnet") return PrepKind::EFFICIENTNET;
    return PrepKind::VGG16;
}

// -------------------- Загрузка модели TFLite --------------------
std::unique_ptr<tflite::Interpreter> load_tflite(
        const std::string& model_path,
        std::unique_ptr<tflite::FlatBufferModel>& model_holder
) {
    model_holder = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_holder) {
        throw std::runtime_error("Не удалось загрузить .tflite модель: " + model_path);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model_holder, resolver)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Не удалось создать интерпретатор TFLite");
    }

    interpreter->SetNumThreads(2);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("AllocateTensors() failed");
    }
    return interpreter;
}

// -------------------- Top-K утилита --------------------
std::vector<std::pair<int, float>> top_k(const std::vector<float>& probs, int k) {
    std::vector<std::pair<int, float>> idx_prob;
    idx_prob.reserve(probs.size());
    for (int i = 0; i < (int)probs.size(); ++i) idx_prob.emplace_back(i, probs[i]);

    if (k > (int)idx_prob.size()) k = (int)idx_prob.size();
    std::partial_sort(
        idx_prob.begin(), idx_prob.begin() + k, idx_prob.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );
    idx_prob.resize(k);
    return idx_prob;
}

// -------------------- Инференс одного изображения --------------------
struct Pred {
    std::string path;
    std::vector<float> probs; // полный softmax
    std::vector<std::pair<int, float>> top3; // (class_index, prob)
};

Pred predict_one(tflite::Interpreter* interp,
                 const std::string& path,
                 const Meta& meta,
                 PrepKind prep_kind)
{
    cv::Mat img_bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        throw std::runtime_error("Не удалось прочитать изображение: " + path);
    }

    // 1) resize_with_pad -> (H,W)
    cv::Mat padded = resize_with_pad_bgr(img_bgr, meta.img_w, meta.img_h);

    // 2) препроцессинг на месте
    switch (prep_kind) {
        case PrepKind::VGG16:
        case PrepKind::RESNET50:
            preprocess_vgg_resnet_inplace_bgr(padded);
            break;
        case PrepKind::EFFICIENTNET:
            preprocess_efficientnet_inplace_rgb(padded);
            break;
        default:
            preprocess_vgg_resnet_inplace_bgr(padded);
    }

    // гарантируем непрерывность памяти перед memcpy
    if (!padded.isContinuous()) padded = padded.clone();

    // 3) записываем в входной тензор [1,H,W,3] (NHWC, float32)
    float* input = interp->typed_input_tensor<float>(0);
    const size_t bytes = static_cast<size_t>(meta.img_w) * meta.img_h * 3 * sizeof(float);

    if (padded.type() != CV_32FC3) {
        throw std::runtime_error("Внутренняя ошибка: ожидался CV_32FC3 после препроцессинга");
    }
    std::memcpy(input, padded.data, bytes);

    // 4) run
    if (interp->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Invoke() failed");
    }

    // 5) читаем выход softmax [1, NUM_CLASSES]
    int out_idx = interp->outputs()[0];
    TfLiteTensor* out_tensor = interp->tensor(out_idx);

    if (out_tensor->type != kTfLiteFloat32) {
        throw std::runtime_error("Выходной тензор не float32 (для quantized модели нужно отдельное чтение)");
    }
    if (out_tensor->dims->size != 2 || out_tensor->dims->data[0] != 1) {
        throw std::runtime_error("Неожиданная форма выхода: ожидается [1, NUM_CLASSES]");
    }

    const float* out = interp->typed_output_tensor<float>(0);
    int num_classes = out_tensor->dims->data[1];

    std::vector<float> probs(num_classes);
    for (int i = 0; i < num_classes; ++i) probs[i] = out[i];

    Pred pred;
    pred.path = path;
    pred.probs = std::move(probs);
    pred.top3 = top_k(pred.probs, 3);
    return pred;
}

std::string label_by_index(const Meta& meta, int idx) {
    if (idx >= 0 && idx < (int)meta.class_names.size()) return meta.class_names[idx];
    return "class_" + std::to_string(idx);
}

void print_top3(const Meta& meta, const Pred& pr) {
    for (int rank = 0; rank < (int)pr.top3.size(); ++rank) {
        int cls = pr.top3[rank].first;
        float p = pr.top3[rank].second;
        std::cout << "  #" << (rank + 1) << ": "
                  << label_by_index(meta, cls)
                  << "  (" << std::fixed << std::setprecision(2) << p * 100.0f << "%)\n";
    }
}

int run_tf()
{
    std::string model_path   = "/home/user/dir/programming/C++/Yaroslava/DIPLOM/data_for_tests/DIPLOM/models/model.tflite";
    std::string classes_json = "/home/user/dir/programming/C++/Yaroslava/DIPLOM/data_for_tests/DIPLOM/classes.json";
    bool batch_mode = false; // (argc >= 4);
    std::string images_dir = batch_mode ? "" /*argv[3]*/ : "";

    try {
        Meta meta = load_meta(classes_json);
        auto prep_kind = select_prep(!meta.preprocess.empty() ? meta.preprocess : meta.base_model);

        std::unique_ptr<tflite::FlatBufferModel> model_holder;
        auto interpreter = load_tflite(model_path, model_holder);

        std::cout << "Модель: " << model_path << "\n";
        std::cout << "Классов: " << meta.class_names.size() << "\n";
        std::cout << "Препроцессинг: " << (meta.preprocess.empty() ? meta.base_model : meta.preprocess) << "\n";
        std::cout << "Размер: " << meta.img_w << "x" << meta.img_h << "\n";
        std::cout << "--------------------------------------------------\n";

        if (batch_mode) {
            auto images = gather_images(images_dir);
            if (images.empty()) {
                std::cout << "В директории нет изображений\n";
                return 0;
            }

            std::cout << "Найдено " << images.size() << " изображений\n";
            std::cout << "--------------------------------------------------\n";

            for (const auto& p : images) {
                auto pr = predict_one(interpreter.get(), p, meta, prep_kind);
                std::cout << fs::path(p).filename().string() << "\n";
                print_top3(meta, pr);
                std::cout << "--------------------------------------------------\n";
            }
            return 0;
        }

        // -------- интерактивный режим: 3 изображения --------
        std::vector<Pred> results;
        results.reserve(3);

        for (int i = 1; i <= 3; ++i) {
            std::cout << "Введите путь к изображению #" << i << " (или 'q' для выхода):\n> ";
            std::string path;
            std::getline(std::cin, path);

            if (path == "q" || path == "Q") {
                std::cout << "Выход.\n";
                // -------- сводка --------
                std::cout << "\n==================== ИТОГИ (топ совпадений по каждой картинке) ====================\n";
                for (size_t i = 0; i < results.size(); ++i) {
                    std::cout << "Изображение #" << (i + 1) << ": " << results[i].path << "\n";
                    print_top3(meta, results[i]);
                    std::cout << "-------------------------------------------------------------------------\n";
                }
                return 0;
            }
            if (path.empty()) {
                std::cout << "Пустой ввод — попробуйте ещё раз.\n";
                --i;
                continue;
            }

            try {
                auto pr = predict_one(interpreter.get(), path, meta, prep_kind);
                results.push_back(pr);

                std::cout << "Результат для #" << i << ":\n";
                print_top3(meta, pr);
                std::cout << "--------------------------------------------------\n";
            } catch (const std::exception& e) {
                std::cout << "Ошибка: " << e.what() << "\n";
                std::cout << "Попробуйте другой файл.\n";
                --i;
            }
        }

        // -------- сводка --------
        std::cout << "\n==================== ИТОГИ (топ-3 по каждой картинке) ====================\n";
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "Изображение #" << (i + 1) << ": " << results[i].path << "\n";
            print_top3(meta, results[i]);
            std::cout << "-------------------------------------------------------------------------\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 2;
    }
}

// ------------------------ main ------------------------

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Использование:\n";
            std::cerr << "  " << argv[0] << " <questions.json> <recognition.json>\n";
            return 1;
        }

        const std::string questionsPath = argv[1];
        const std::string recognitionPath = argv[2];

        const json questionsTree = JsonFileLoader::LoadFromFile(questionsPath);
        const json recognitionJson = JsonFileLoader::LoadFromFile(recognitionPath);

        ConsoleUI ui;
        RecognitionCatalog catalog(recognitionJson);
        DecisionTreeEngine engine(questionsTree, catalog, ui);

        engine.Run();
        run_tf();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << '\n';
        return 1;
    }
}