// Signs a release's SHA256SUMS with the Ed25519 key in $MLXRUNNER_UPDATE_KEY (base64 of the raw
// 32-byte private key), writing SHA256SUMS.sig (raw 64-byte signature). Used by the release workflow.
//
//   MLXRUNNER_UPDATE_KEY=… swift scripts/sign.swift dist/release/SHA256SUMS
//
// The signature is then checked against packaging/update-key.pub, so a wrong or rotated secret
// fails the release instead of shipping updates nobody can install.
import CryptoKit
import Foundation

let args = CommandLine.arguments
guard args.count == 2 else { fatalError("usage: swift scripts/sign.swift <SHA256SUMS>") }
guard let secret = ProcessInfo.processInfo.environment["MLXRUNNER_UPDATE_KEY"],
      let raw = Data(base64Encoded: secret.trimmingCharacters(in: .whitespacesAndNewlines)) else {
    fatalError("MLXRUNNER_UPDATE_KEY is not set: releases must be signed")
}
let key = try Curve25519.Signing.PrivateKey(rawRepresentation: raw)
let file = URL(fileURLWithPath: args[1])
let data = try Data(contentsOf: file)
let sig = try key.signature(for: data)

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let pem = try String(contentsOf: root.appendingPathComponent("packaging/update-key.pub"), encoding: .utf8)
let b64 = pem.split(separator: "\n").filter { !$0.hasPrefix("-----") }.joined()
guard let spki = Data(base64Encoded: b64), spki.count == 44,
      let pub = try? Curve25519.Signing.PublicKey(rawRepresentation: spki.suffix(32)),
      pub.isValidSignature(sig, for: data) else {
    fatalError("signature doesn't verify with packaging/update-key.pub: wrong signing key")
}
try sig.write(to: file.appendingPathExtension("sig"))
print("signed   \(file.lastPathComponent).sig (verified with packaging/update-key.pub)")
