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
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << '\n';
        return 1;
    }
}