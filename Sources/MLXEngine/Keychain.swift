import Foundation
import Security

/// Minimal generic-password Keychain wrapper (login keychain, Security.framework only).
public enum Keychain {
    public static let defaultService = "de.21no.mlxrunner"

    private static func query(service: String, account: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }

    /// Returns the stored string, or nil when the item does not exist or cannot be read.
    public static func read(account: String, service: String = defaultService) -> String? {
        var q = query(service: service, account: account)
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Creates or replaces the item. An empty value deletes it.
    @discardableResult
    public static func write(_ value: String, account: String, service: String = defaultService) -> Bool {
        if value.isEmpty { return delete(account: account, service: service) }
        let data = Data(value.utf8)
        let q = query(service: service, account: account)
        let status = SecItemUpdate(q as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecSuccess { return true }
        guard status == errSecItemNotFound else { return false }
        var add = q
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }

    @discardableResult
    public static func delete(account: String, service: String = defaultService) -> Bool {
        let status = SecItemDelete(query(service: service, account: account) as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
}

/// Where the Hugging Face access token lives. `HF_TOKEN` in the environment always wins;
/// otherwise the Keychain (service "de.21no.mlxrunner", account "hf-token"). A token left in
/// UserDefaults by older builds is migrated into the Keychain on first read and removed.
public enum HFToken {
    public static let account = "hf-token"
    static let legacyDefaultsKey = "hfToken"

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cached: String??

    /// The token to send, or nil.
    public static var current: String? {
        if let env = ProcessInfo.processInfo.environment["HF_TOKEN"], !env.isEmpty { return env }
        return stored
    }

    /// True when `HF_TOKEN` overrides the stored token.
    public static var isFromEnvironment: Bool {
        !(ProcessInfo.processInfo.environment["HF_TOKEN"] ?? "").isEmpty
    }

    /// The token saved in the Keychain (after migrating any legacy UserDefaults value).
    public static var stored: String? {
        lock.lock(); defer { lock.unlock() }
        if let c = cached { return c }
        var token = Keychain.read(account: account)
        let defaults = UserDefaults.standard
        if let legacy = defaults.string(forKey: legacyDefaultsKey) {
            if token == nil, !legacy.isEmpty, Keychain.write(legacy, account: account) { token = legacy }
            // Only drop the plaintext copy once the Keychain holds a token (or the legacy value was empty).
            if token != nil || legacy.isEmpty { defaults.removeObject(forKey: legacyDefaultsKey) }
        }
        let value = (token?.isEmpty ?? true) ? nil : token
        cached = .some(value)
        return value
    }

    /// Saves (or, when empty, deletes) the token in the Keychain.
    @discardableResult
    public static func save(_ token: String) -> Bool {
        let t = token.trimmingCharacters(in: .whitespacesAndNewlines)
        lock.lock(); defer { lock.unlock() }
        let ok = Keychain.write(t, account: account)
        cached = ok ? .some(t.isEmpty ? nil : t) : nil
        UserDefaults.standard.removeObject(forKey: legacyDefaultsKey)
        return ok
    }
}
