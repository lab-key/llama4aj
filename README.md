# llama4aj

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## About

A Direct Android & Java Build For The [mybigday](https://github.com/mybigday) Serverless Server-like Chat Completion Implementation

Java 11 Required

Supports Android, Linux - Currently tested

In theory should support Windows, BSD & macOS - Still needs to be tested!

Allows For Fast Inference & Intuitive App Building - Checkout The Example Apps & Try Building Them Yourself!

## Getting Started

```
./gradlew :examples:android-app:build # You can just copy paste the .apk file to your phone and install it.
# Or installDebug If You Have adb
#
# Or For The More Simple Desktop App
# mv model.gguf examples/desktop-app/ - Place The Model inside examples/desktop-app/
./gradlew :examples:desktop-app:run
```

## Example
```
import com.llama4aj;
```

***COMING TO MAVEN SOON***

***SCALA VERSION COMING SOON***

## TODO

A lot more configuring to make sure it is alligned with upstream [llama.rn](https://github.com/mybigday/llama.rn) - things like cpu variants ( Just properly defining and doing everything what the upstream projects provide - still bits TODO but getting close to 0.0.1 - alpha )

Proper syncing system

Cleanup...

Tests

Especially tests on other platforms and architectures if you have time, knowhow and want to help would be appreciated!

Docs

Videos

Cleanup........

[mybigday]:(https://github.com/mybigday)
