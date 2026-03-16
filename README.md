# llama4aj

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## About

A Direct Android & Java Build For The [mybigday](https://github.com/mybigday) Serverless Server-like Chat Completion Implementation

Java 11 Required

Supports Android, Linux - Currently tested

In theory should support Windows, BSD & macOS - Still needs to be tested!

Allows For Fast Inference & Intuitive App Building

## Getting Started

```
git clone https://github.com/ForbiddenByte/llama4aj.git

cd llama4aj

./gradlew :examples:android-app:build
# You can just copy paste the .apk file to your phone and install it.
# Or installDebug If You Have adb
#
# Or For The More Simple Desktop App
# Place The Model inside examples/desktop-app/ - mv model.gguf examples/desktop-app/
./gradlew :examples:desktop-app:run
```

##### Simple Example
```java
import com.llama4aj;

public class ChatExample {
    public static void main(String[] args) {
        llama4aj.generate("model.gguf", "Hello!", System.out::print);
    }
}
// Expects the model to be in the same directory / folder
```

Or make your own offline / local ChatGPT clone with 500 lines of code!

Check the Desktop App for the very simple version

***COMING TO MAVEN SOON***

***SCALA VERSION COMING SOON***

## TODO

A lot more configuring to make sure it is alligned with upstream [llama.rn](https://github.com/mybigday/llama.rn) - things like cpu variants ( Just properly defining and doing everything what the upstream projects provide - still bits TODO but getting close to 0.0.1 - alpha )

Implement in different Java versions

Proper syncing system

Cleanup...

Tests

Especially tests on other platforms and architectures if you have time, knowhow and want to help would be appreciated!

Docs

Videos

Cleanup........

[mybigday]:(https://github.com/mybigday)
