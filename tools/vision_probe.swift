import Foundation
import Vision
import CoreGraphics
import ImageIO

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

let args = CommandLine.arguments
guard args.count == 2 else {
    fputs("usage: vision_probe.swift <image_path>\n", stderr)
    throw ProbeError.invalidArgs
}

let imageURL = URL(fileURLWithPath: args[1])
let image = try loadImage(url: imageURL)
let result = try performProbe(on: image)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(result)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write("\n".data(using: .utf8)!)
