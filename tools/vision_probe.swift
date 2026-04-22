import Foundation
import Vision
import CoreGraphics
import ImageIO

struct ProbeEnvelope: Codable {
    let ok: Bool
    let result: ProbeResult?
    let error: String?
}

struct Classification: Codable {
    let identifier: String
    let confidence: Double
}

struct ProbeResult: Codable {
    let labels: [Classification]
    let text: [String]
    let faceCount: Int
    let bestFaceCaptureQuality: Double?
    let largestFaceArea: Double
}

enum ProbeError: Error {
    case invalidArgs
    case imageLoadFailed
}

func loadImage(url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw ProbeError.imageLoadFailed
    }
    return image
}

func performProbe(on image: CGImage) throws -> ProbeResult {
    let handler = VNImageRequestHandler(cgImage: image)

    let classify = VNClassifyImageRequest()
    classify.usesCPUOnly = false

    let recognizeText = VNRecognizeTextRequest()
    recognizeText.recognitionLevel = .fast
    recognizeText.recognitionLanguages = ["zh-Hans", "en-US"]
    recognizeText.usesLanguageCorrection = false

    let faceQuality = VNDetectFaceCaptureQualityRequest()

    try handler.perform([classify, recognizeText, faceQuality])

    let labels = (classify.results ?? [])
        .prefix(12)
        .map { Classification(identifier: $0.identifier, confidence: Double($0.confidence)) }

    let textLines = (recognizeText.results ?? [])
        .compactMap { observation in observation.topCandidates(1).first?.string }

    let faces = faceQuality.results ?? []
    let bestFaceCaptureQuality = faces
        .compactMap { $0.faceCaptureQuality.map(Double.init) }
        .max()
    let largestFaceArea = faces
        .map { Double($0.boundingBox.width * $0.boundingBox.height) }
        .max() ?? 0.0

    return ProbeResult(
        labels: labels,
        text: textLines,
        faceCount: faces.count,
        bestFaceCaptureQuality: bestFaceCaptureQuality,
        largestFaceArea: largestFaceArea
    )
}

func writeJSON<T: Encodable>(_ value: T, pretty: Bool) throws {
    let encoder = JSONEncoder()
    if pretty {
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    }
    let data = try encoder.encode(value)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

func runSingle(imagePath: String) throws {
    try autoreleasepool {
        let imageURL = URL(fileURLWithPath: imagePath)
        let image = try loadImage(url: imageURL)
        let result = try performProbe(on: image)
        try writeJSON(result, pretty: true)
    }
}

func runStdio() {
    while let rawLine = readLine(strippingNewline: true) {
        let imagePath = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        if imagePath.isEmpty {
            continue
        }
        do {
            try autoreleasepool {
                let imageURL = URL(fileURLWithPath: imagePath)
                let image = try loadImage(url: imageURL)
                let result = try performProbe(on: image)
                try writeJSON(
                    ProbeEnvelope(ok: true, result: result, error: nil),
                    pretty: false
                )
            }
        } catch {
            let message = String(describing: error)
            try? writeJSON(
                ProbeEnvelope(ok: false, result: nil, error: message),
                pretty: false
            )
        }
    }
}

let args = Array(CommandLine.arguments.dropFirst())
if args == ["--stdio"] {
    runStdio()
} else if args.count == 1 {
    try runSingle(imagePath: args[0])
} else {
    fputs("usage: vision_probe.swift <image_path> | --stdio\n", stderr)
    throw ProbeError.invalidArgs
}
